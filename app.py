import os
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from train_model import MODEL_FILE

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / MODEL_FILE
DATABASE = BASE_DIR / "users.db"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key-before-production")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

FEATURES = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]
NUMERIC = ["Age", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]
LABELS = {0: "Lower model-indicated risk", 1: "Higher model-indicated risk"}


def init_db():
    con = sqlite3.connect(DATABASE)
    con.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)")
    con.commit(); con.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_bundle():
    return joblib.load(MODEL_PATH)

def row_from_form(form):
    return {
        "Age": float(form["Age"]), "Gender": form["Gender"],
        "TB": float(form["TB"]), "DB": float(form["DB"]),
        "Alkphos": float(form["Alkphos"]), "Sgpt": float(form["Sgpt"]),
        "Sgot": float(form["Sgot"]), "TP": float(form["TP"]),
        "ALB": float(form["ALB"]), "A/G Ratio": float(form["AG"])
    }


def validate_row(row):
    if not (0 <= row["Age"] <= 120): raise ValueError("Age must be between 0 and 120.")
    for k in NUMERIC[1:]:
        if row[k] < 0: raise ValueError(f"{k} cannot be negative.")


def predict_row(row, bundle):
    df = pd.DataFrame([row], columns=FEATURES)
    x = bundle["preprocessor"].transform(df)
    # Sparse one-hot output is converted to dense because the 4-qubit quantum model expects an array.
    if hasattr(x, "toarray"): x = x.toarray()
    x4 = bundle["pca"].transform(x)
    angles = bundle["angle_scaler"].transform(x4)

    q_pred = int(np.asarray(bundle["vqc"].predict(angles)).ravel()[0])
    q_probs = np.asarray(bundle["vqc"].predict_proba(angles))[0]
    q_score = float(q_probs[1]) * 100.0

    c_pred = int(bundle["classical"].predict(x)[0])
    c_probs = bundle["classical"].predict_proba(x)[0]
    c_score = float(c_probs[1]) * 100.0

    return {
        "quantum_pred": q_pred, "quantum_label": LABELS[q_pred], "quantum_score": q_score,
        "classical_pred": c_pred, "classical_label": LABELS[c_pred], "classical_score": c_score,
        "agreement": q_pred == c_pred
    }


def what_if(row, bundle, baseline):
    rows=[]
    for feature in NUMERIC[1:]:
        base=float(row[feature])
        if base == 0:
            scenarios=[("+10%",0.0), ("-10%",0.0)]
        else:
            scenarios=[("+10%",base*1.10),("-10%",max(0,base*0.90))]
        for label,val in scenarios:
            changed=dict(row); changed[feature]=val
            p=predict_row(changed,bundle)
            rows.append({"feature":feature,"change":label,"new_value":val,"pred":p["quantum_pred"],"label":p["quantum_label"],"score":p["quantum_score"],"delta":p["quantum_score"]-baseline["quantum_score"],"transition":p["quantum_pred"] != baseline["quantum_pred"]})
    return rows

@app.route("/")
def home():
    return redirect(url_for("dashboard" if "username" in session else "login"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        username=request.form.get("username","").strip(); password=request.form.get("password","")
        if not username or not password: flash("Enter both username and password."); return redirect(url_for("register"))
        con=sqlite3.connect(DATABASE)
        try:
            con.execute("INSERT INTO users (username,password_hash) VALUES (?,?)",(username,generate_password_hash(password)))
            con.commit(); flash("Registration successful. Please log in."); return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("That username already exists."); return redirect(url_for("register"))
        finally: con.close()
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        username=request.form.get("username","").strip(); password=request.form.get("password","")
        con=sqlite3.connect(DATABASE); row=con.execute("SELECT username,password_hash FROM users WHERE username=?",(username,)).fetchone(); con.close()
        if row and check_password_hash(row[1],password): session["username"]=username; return redirect(url_for("dashboard"))
        flash("Invalid username or password.")
    return render_template("login.html")

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "username" not in session: return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    if "username" not in session: return redirect(url_for("login"))
    if request.method=="POST":
        try:
            row=row_from_form(request.form); validate_row(row); bundle=get_bundle(); result=predict_row(row,bundle); scenarios=what_if(row,bundle,result)
            transitions=[s for s in scenarios if s["transition"] and s["pred"] == 1]
            return render_template("result.html",row=row,result=result,scenarios=scenarios,transitions=transitions)
        except Exception as e:
            flash(f"Prediction could not be completed: {e}"); return redirect(url_for("predict"))
    return render_template("predict.html")

@app.route("/upload", methods=["GET","POST"])
def upload():
    if "username" not in session: return redirect(url_for("login"))
    if request.method=="POST":
        f=request.files.get("report")
        if not f or f.filename=="": flash("Select an LFT report image."); return redirect(url_for("upload"))
        if not allowed_file(f.filename): flash("Use PNG, JPG or JPEG."); return redirect(url_for("upload"))
        name=secure_filename(f.filename); f.save(UPLOAD_FOLDER/name)
        return render_template("upload_result.html",filename=name)
    return render_template("upload.html")

@app.route("/health")
def health(): return {"status":"ok"}

init_db()

if __name__=="__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )