from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from model_utils import parity_interpret

BASE_DIR=Path(__file__).resolve().parent
MODEL_DIR=BASE_DIR/"models"; MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE="models/model_bundle.joblib"
REPORT_FILE="models/training_report.json"
FEATURES=["Age","Gender","TB","DB","Alkphos","Sgpt","Sgot","TP","ALB","A/G Ratio"]
NUMERIC=["Age","TB","DB","Alkphos","Sgpt","Sgot","TP","ALB","A/G Ratio"]
CATEGORICAL=["Gender"]


def load_data():
    ds=fetch_ucirepo(id=225)
    X=ds.data.features.copy(); y=ds.data.targets.copy()
    # UCI exposes the ten predictors; normalize names for this project.
    X.columns=FEATURES
    y=y.iloc[:,0].astype(int).map({1:1,2:0})
    return X,y


def preprocess():
    num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    return ColumnTransformer([("num",num,NUMERIC),("cat",cat,CATEGORICAL)])


def balance(X,y,seed=42):
    rng=np.random.default_rng(seed); y=np.asarray(y)
    ids=[]
    classes,counts=np.unique(y,return_counts=True); target=max(counts)
    for c,n in zip(classes,counts):
        cids=np.where(y==c)[0]
        ids.extend(rng.choice(cids,target,replace=(n<target)))
    ids=np.array(ids); rng.shuffle(ids)
    return X[ids], y[ids]


def metrics(y,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall_sensitivity":float(recall_score(y,pred,zero_division=0)),"specificity":float(tn/(tn+fp) if (tn+fp) else 0),"f1":float(f1_score(y,pred,zero_division=0))}



def train_and_save():
    print("Downloading/loading UCI ILPD dataset...")
    X,y=load_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
    prep=preprocess(); Xt=prep.fit_transform(X_train); Xv=prep.transform(X_test)
    Xt_bal,yb=balance(Xt,y_train.to_numpy())

    # Classical baseline: SVM on the same preprocessed representation.
    classical=SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42)
    classical.fit(Xt_bal,yb)
    c_pred=classical.predict(Xv)

    # Hybrid quantum pipeline: PCA reduces the preprocessed vector to four quantum features.
    pca=PCA(n_components=4,random_state=42)
    Zt=pca.fit_transform(Xt_bal); Zv=pca.transform(Xv)
    angle_scaler=MinMaxScaler(feature_range=(-np.pi,np.pi))
    At=angle_scaler.fit_transform(Zt); Av=angle_scaler.transform(Zv)

    feature_map=ZZFeatureMap(feature_dimension=4,reps=1)
    ansatz=RealAmplitudes(num_qubits=4,reps=1)
    sampler=StatevectorSampler()
    vqc=VQC(feature_map=feature_map,ansatz=ansatz,sampler=sampler,optimizer=COBYLA(maxiter=40),interpret=parity_interpret,output_shape=2)
    print("Training 4-qubit VQC. This can take several minutes on a normal laptop...")
    vqc.fit(At,yb)
    q_pred=np.asarray(vqc.predict(Av)).ravel().astype(int)

    report={
      "dataset":"UCI ILPD (id=225)","rows":int(len(X)),"features":FEATURES,
      "split":"80/20 stratified; random_state=42","quantum_qubits":4,
      "quantum_model":"VQC with ZZFeatureMap + RealAmplitudes + StatevectorSampler + COBYLA",
      "classical_model":"RBF SVM",
      "classical_metrics":metrics(y_test,c_pred),"quantum_metrics":metrics(y_test,q_pred),
      "note":"Metrics are from this fixed split and are not clinical validation. The app is a screening/research demonstration."
    }
    bundle={"preprocessor":prep,"pca":pca,"angle_scaler":angle_scaler,"classical":classical,"vqc":vqc,"features":FEATURES}
    joblib.dump(bundle,BASE_DIR/MODEL_FILE)
    (BASE_DIR/REPORT_FILE).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    return bundle,report


def ensure_models():
    if not (BASE_DIR/MODEL_FILE).exists(): return train_and_save()[0]
    return joblib.load(BASE_DIR/MODEL_FILE)

if __name__=="__main__": train_and_save()
