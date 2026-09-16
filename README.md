# Hybrid Quantum Liver Disease Early-Risk Screening — SIH26139 Prototype

This project upgrades the original Flask liver-disease website into a **hybrid quantum-classical machine-learning prototype**.

## What it does
1. Uses the UCI Indian Liver Patient Dataset (ILPD, dataset id 225) for training.
2. Performs classical preprocessing: imputation, scaling and one-hot encoding.
3. Uses PCA to reduce the preprocessed data to **4 features / 4 qubits**.
4. Scales the four components to quantum rotation angles.
5. Trains a genuine **Variational Quantum Classifier (VQC)** with Qiskit.
6. Trains an RBF-SVM baseline on the same preprocessed data.
7. Compares the quantum and classical predictions and reports accuracy, sensitivity, specificity and F1.
8. The website accepts the ten input values used by the ILPD dataset and shows:
   - the entered LFT values,
   - quantum-model classification,
   - quantum model score (NOT a clinical probability),
   - classical baseline result,
   - agreement/disagreement between models,
   - a what-if table showing how a simulated +/-10% change in each laboratory value changes the model output.

## Important medical limitation
This is an **academic early-risk screening/research prototype**. It does not diagnose liver disease and it does not replace a doctor or a laboratory's reference ranges. A model score is not a medical probability. Do not make treatment decisions from this website.

## Dataset
The UCI ILPD contains 583 records and 10 predictor variables: Age, Gender, total bilirubin, direct bilirubin, alkaline phosphatase, SGPT/ALT, SGOT/AST, total proteins, albumin and A/G ratio. UCI reports 416 liver-disease-class records and 167 non-liver-disease-class records. Source: https://archive.ics.uci.edu/dataset/225/ilpd+indian+

The training script downloads the dataset through `ucimlrepo`.

## 1. Install Python
Use Python 3.14.7 for the supplied environment. The dependency pins have been updated for Python 3.14 (including pandas 2.3.3).

Check:
```bash
python --version
```

## 2. Open the project in VS Code
Extract the ZIP, then open the folder:
`Hybrid_Quantum_Liver_SIH26139`

## 3. Create a virtual environment (recommended)
Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, open Command Prompt and use:
```cmd
.venv\Scripts\activate
```

## 4. Install packages
```bash
pip install -r requirements.txt
```

Qiskit Machine Learning 0.9.1 currently documents Qiskit 2.5.2 and the VQC API used here.

## 5. Train the models
Run this once:
```bash
python train_model.py
```
The script downloads ILPD, trains the classical SVM and the 4-qubit VQC, evaluates both on the held-out test set, and saves:
- `models/model_bundle.joblib`
- `models/training_report.json`

Training can take several minutes because the quantum classifier is optimized iteratively on a simulator.

## 6. Start the website
```bash
python app.py
```
Open:
`http://127.0.0.1:5000`

Register -> Login -> Enter LFT Values -> Predict.

## 7. How to use an existing LFT report
The Upload Report page lets you store an image of the report, but the app intentionally does **not** blindly OCR medical values. Read the report and enter the verified values into the form. This avoids an OCR typo becoming a false medical-looking result.

The project uses the ILPD dataset's feature definitions. Laboratory units/reference ranges can differ between laboratories, so enter values according to the dataset/input convention used by the project and explain this limitation during your demo.

## 8. Understanding the result
### Higher model-indicated risk
The quantum model classified the input into the liver-disease class learned from ILPD.

### Lower model-indicated risk
The quantum model classified the input into the non-liver-disease class learned from ILPD.

### Quantum model score
This is the VQC output for the disease class. It is a **model score**, not a clinically calibrated probability.

### Classical comparison
The RBF-SVM gives an independent classical baseline. Showing both supports the hybrid-QML requirement and prevents the presentation from claiming that quantum ML is automatically better.

### What-if table
Each row changes one numeric input by +10% or -10% and runs the model again. This is a **sensitivity demonstration**, not a medical recommendation or a clinical threshold test.

## 9. SIH26139 positioning
For the presentation, explain the pipeline as:

LFT/clinical inputs -> classical preprocessing -> PCA -> quantum feature encoding -> 4-qubit VQC -> early-risk classification
                                                                           -> classical SVM baseline -> comparison

Do not claim that the quantum model is superior unless the saved test metrics actually show that.

## 10. Deploy to Render
1. Put this project in a GitHub repository.
2. In Render choose **New -> Web Service** and connect the repository.
3. Runtime: Python 3.
4. Build command:
```text
pip install -r requirements.txt
```
5. Start command:
```text
gunicorn app:app
```
6. Set `SECRET_KEY` to a long random value in Render Environment Variables.
7. Deploy.

### Important deployment note
The first deployment will need to install Qiskit and may need to train the VQC if `models/model_bundle.joblib` is not already present. For a hackathon demo, train locally first and keep the generated model artifact available to the deployment process. Do not put patient reports or private medical data into a public Git repository.

## 11. Troubleshooting
- `ModuleNotFoundError: qiskit`: run `pip install -r requirements.txt` inside the active virtual environment.
- `model_bundle.joblib not found`: run `python train_model.py`.
- Training is slow: reduce `COBYLA(maxiter=40)` for a demonstration, but record the exact setting used for your evaluation.
- Render Python version mismatch: use the included `.python-version` or set `PYTHON_VERSION` to `3.14.7`.

## Suggested SIH demo statement
"Our system combines classical preprocessing and dimensionality reduction with a four-qubit Variational Quantum Classifier for early liver-disease risk screening. We benchmark the quantum model against an RBF-SVM baseline and expose a what-if analysis so a user can see how simulated changes in the entered laboratory values affect the model output. The system is a screening/research prototype and does not replace clinical diagnosis."
