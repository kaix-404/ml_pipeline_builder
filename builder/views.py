from django.shortcuts import render
from django.http import HttpResponse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix

import json
import io, base64
import joblib

def home(request):
    return render(request, 'home.html')

def upload_page(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("dataset")
        if not uploaded_file:
            return render(request, "partials/upload_result.html", {"error": "No file uploaded."})
        try:
            if uploaded_file.name.endswith(".csv"):
                encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
                raw = uploaded_file.read()
                df = None
                for enc in encodings:
                    try:
                        text = raw.decode(enc, errors="ignore")
                        df = pd.read_csv(io.StringIO(text))
                        break
                    except Exception:
                        df = None
                if df is None:
                    return render(request, "partials/upload_result.html", {"error": "Could not read CSV file with standard encodings."})
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                return render(request, "partials/upload_result.html", {"error": "Unsupported file format"})
        except Exception as e:
            return render(request, "partials/upload_result.html", {"error": f"Error reading file: {e}"})
        request.session["df_json"] = df.to_json()
        request.session["filename"] = uploaded_file.name
        return render(request, "partials/upload_result.html", {
            "filename": uploaded_file.name,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": list(df.columns),
        })
    return render(request, "upload.html")

def preprocess_page(request):
    df_json = request.session.get("df_json")
    filename = request.session.get("filename")
    df = pd.read_json(df_json) if df_json else None
    return render(request, "preprocess.html", {"filename": filename, "df": df})

def extract_text(request):
    df_json = request.session.get("df_json")
    if not df_json:
        return HttpResponse("<p>No data uploaded.</p>")
    try:
        df = pd.read_json(df_json)
        preview_df = df.head(5)
        text = preview_df.to_string(index=False)
    except Exception as e:
        return HttpResponse(f"<p>Error reading data: {e}</p>")
    request.session["raw_text"] = text
    return HttpResponse(f"<h3>Extracted Text:</h3><pre>{text}</pre>")

def standardize(request):
    df_json = request.session.get("df_json")
    if not df_json:
        return HttpResponse("<p>No data uploaded.</p>")
    df = pd.read_json(df_json)
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    except Exception as e:
        return HttpResponse(f"<p>Error during standardization: {e}</p>")
    request.session["df_json"] = df.to_json()
    return HttpResponse(f"<h3>Standardization applied on numeric columns:</h3><pre>{', '.join(numeric_cols)}</pre>")

def normalize(request):
    df_json = request.session.get("df_json")
    if not df_json:
        return HttpResponse("<p>No data uploaded.</p>")
    df = pd.read_json(df_json)
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    except Exception as e:
        return HttpResponse(f"<p>Error during normalization: {e}</p>")
    request.session["df_json"] = df.to_json()
    return HttpResponse(f"<h3>Normalization applied on numeric columns:</h3></pre>{', '.join(numeric_cols)}</pre>")

def split_page(request):
    return render(request, "split.html")

def train_test_split_view(request):
    df_json = request.session.get("df_json")
    if not df_json:
        return HttpResponse("<p>No data uploaded.</p>")
    df = pd.read_json(df_json)
    if df.shape[1] < 2:
        return HttpResponse("<p>Not enough columns for ML.</p>")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    train_custom = request.POST.get("train_custom")
    test_custom = request.POST.get("test_custom")
    if train_custom and test_custom:
        try:
            train_p = float(train_custom) / 100
            test_p = float(test_custom) / 100
            if abs((train_p + test_p) - 1.0) > 0.001:
                return HttpResponse("<p>Train + Test must equal 100%.</p>")
            test_size = test_p
        except:
            return HttpResponse("<p>Invalid custom percentages.</p>")
    else:
        test_size = float(request.POST.get("ratio", 0.2))
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    except Exception as e:
        return HttpResponse(f"<p>Error during train-test split: {e}</p>")
    request.session["X_train"] = X_train.to_json()
    request.session["X_test"] = X_test.to_json()
    request.session["y_train"] = y_train.to_frame().to_json()
    request.session["y_test"] = y_test.to_frame().to_json()
    percent = round((1 - test_size) * 100)
    return render(request, "partials/split_result.html", {
        "train": percent,
        "test": round(test_size * 100)
    })

def train_model(request):
    X_train_json = request.session.get("X_train")
    X_test_json = request.session.get("X_test")
    y_train_json = request.session.get("y_train")
    y_test_json = request.session.get("y_test")

    if not all([X_train_json, X_test_json, y_train_json, y_test_json]):
        return HttpResponse("<p>Train/Test data not found. Please split data first.</p>")

    X_train = pd.read_json(X_train_json)
    X_test = pd.read_json(X_test_json)
    y_train = pd.read_json(y_train_json).iloc[:, 0]
    y_test = pd.read_json(y_test_json).iloc[:, 0]

    X_train = X_train.fillna(X_train.mean(numeric_only=True))
    X_test = X_test.fillna(X_train.mean(numeric_only=True))

    model_type = request.POST.get("model", "linear")
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    else:
        return HttpResponse("<p>Invalid model selected.</p>")

    for col in X_train.columns:
        if X_train[col].dtype == "object":
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
            le.fit(combined)

            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if model_type == "linear":
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            request.session["reg_metrics"] = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            report = ""
            acc = None
            y_proba = None

        elif model_type == "logistic":
            y_proba = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

        else:
            y_proba = None
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

    except Exception as e:
        return HttpResponse(f"<p>Error training model: {e}</p>")

    request.session["y_pred"] = json.dumps(y_pred.tolist())
    request.session["y_test"] = json.dumps(y_test.tolist())
    request.session["accuracy"] = acc
    request.session["report"] = report
    request.session["model_type"] = model_type

    if model_type == "logistic":
        request.session["y_proba"] = json.dumps(y_proba.tolist())

    if model_type == "linear":
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        request.session["mse"] = mse
        request.session["mae"] = mae
        request.session["r2"] = r2

    joblib.dump(model, "final_model.pkl")

    return render(request, "partials/model_result.html", {
        "model_type": model_type,
        "accuracy": acc,
        "report": report,
        "mse": request.session.get("mse"),
        "mae": request.session.get("mae"),
        "r2": request.session.get("r2")
    })

def results(request):
    y_test_json = request.session.get("y_test")
    y_pred_json = request.session.get("y_pred")
    accuracy = request.session.get("accuracy")
    model_type = request.session.get("model_type")
    X_test_json = request.session.get("X_test")

    if not y_test_json or not y_pred_json:
        return HttpResponse("Prediction data missing.")

    y_test = np.array(json.loads(y_test_json)).ravel()
    y_pred = np.array(json.loads(y_pred_json)).ravel()

    if model_type == "linear":
        metrics = request.session.get("reg_metrics")

        residuals = y_test - y_pred

        fig_res, ax_res = plt.subplots(figsize=(5, 4))
        ax_res.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        ax_res.axhline(y=0, color='r', linestyle='--', lw=2)
        ax_res.set_xlabel("Predicted Values")
        ax_res.set_ylabel("Residuals")
        ax_res.set_title("Residual Plot")
        ax_res.grid(alpha=0.3)

        buf_res = io.BytesIO()
        plt.savefig(buf_res, format="png")
        plt.close(fig_res)
        cm_image = base64.b64encode(buf_res.getvalue()).decode()

        fig_reg, ax_reg = plt.subplots(figsize=(5, 4))
        ax_reg.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        ax_reg.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")
        ax_reg.set_xlabel("Actual Values")
        ax_reg.set_ylabel("Predicted Values")
        ax_reg.set_title("Linear Regression: Actual vs Predicted")
        ax_reg.legend()
        ax_reg.grid(alpha=0.3)

        buf_reg = io.BytesIO()
        plt.savefig(buf_reg, format="png")
        plt.close(fig_reg)
        graph_image = base64.b64encode(buf_reg.getvalue()).decode()

        return render(request, "results.html", {
            "model_type": model_type,
            "metrics": metrics,
            "graph_image": graph_image,
            "cm_image": cm_image,
        })

    cm = confusion_matrix(y_test, y_pred)

    fig1, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png")
    plt.close(fig1)
    cm_image = base64.b64encode(buf1.getvalue()).decode()

    X_test = pd.read_json(X_test_json)
    fig2 = None

    if model_type == "logistic":
        y_proba_json = request.session.get("y_proba")
        if not y_proba_json:
            return HttpResponse("<p>No probability data. Retrain model.</p>")
        y_proba = np.array(json.loads(y_proba_json))

        x_vals = np.arange(len(y_proba))

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(x_vals, y_proba, label="Sigmoid Probability")
        ax2.scatter(x_vals, y_test, color="red", s=10, label="Actual Class")
        ax2.set_title("Logistic Regression Sigmoid Curve")
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()

    elif model_type == "decision_tree":
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            x = X_test[numeric_cols[0]]
            y = X_test[numeric_cols[1]]
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.scatter(x, y, c=y_pred, cmap="coolwarm")
            ax2.set_title("Decision Tree Split Visualization")

    graph_image = ""
    if fig2:
        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png")
        plt.close(fig2)
        graph_image = base64.b64encode(buf2.getvalue()).decode()

    return render(request, "results.html", {
        "model_type": model_type,
        "accuracy": accuracy,
        "cm_image": cm_image,
        "graph_image": graph_image,
    })
