# No-Code ML Pipeline Builder

### 🌐 Live Demo: https://no-code-ml-cnqq.onrender.com

A fully interactive no-code machine learning workflow builder built using **Django + HTMX + Tailwind CSS**.

Upload a dataset → preprocess → split → train → visualize → get results - all without writing a single line of Python.

---

## ⭐ Features

- **Dataset Upload**
    - Upload CSV/Excel datasets
    - Live preview of the first rows
    - Automatic datatype detection
    
- **Preprocessing Tools**
    - Extract first 5 rows
    - Standardization
    - Normalization
    - Missing value handling
    - HTMX-powered instant updates

- **Train–Test Split**
    - Choose split ratio (80/20, 75/25, 70/30 or custom input) 
    - Stores split in session for next steps
      
- **Model Training**
    - Supports multiple ML models:
    - Linear Regression
    - Logistic Regression
    - Decision Tree Classifier
    - Includes:
        - Proper categorical encoding
        - Consistent label transformation
        - Session-based model persistence
        - Automatic saving as ```final_model.pkl```
          
- **Results & Visualization**
    - Confusion matrix heatmap
    - Sigmoid curve for Logistic Regression
    - Regression line for Linear Regression
    - Scatter-based split visualization for Decision Trees
    - Classification report
    - Accuracy / RMSE / R² depending on model

- **UI / UX Enhancements**
    - Midnight glassmorphism theme
    - Animated starry sky background
    - Gradient buttons with glow effects
    - Smooth navigation with HTMX

---

## 🛠️ Tech Stack

| Layer            | Technology                             |
|------------------|----------------------------------------|
| Frontend         | HTMX, Tailwind CSS                     |
| Backend          | Django 5, Python 3                     |
| ML Engine        | scikit-learn, NumPy, Pandas            |
| Visualization    | Matplotlib, Seaborn                    |

---

## 📂 Project Structure

```csharp
ml_pipeline_builder/
│── .venv/                      # Virtual environment (ignored)
│── builder/                    # Main Django app
│   ├── migrations/
│   ├── templates/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
│
│── ml_pipeline_builder/        # Django project folder
│── db.sqlite3                  # Local database
│── final_model.pkl             # Saved ML model
│── manage.py                   # Django runner
│── .gitignore
│── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kaix-404/ml_pipeline_builder.git
cd ml_pipeline_builder
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run The Server

```bash
python manage.py runserver
```

Then open: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 🧪 How To Use

- Upload dataset
- Choose preprocessing options
- Split dataset
- Pick a model
- View results:
    - Confusion matrix
    - Curves
    - Accuracy / errors
    - Classification report

---

## 📜 Requirements File

Make sure to include this:

```nginx
Django
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
openpyxl
```

---

## ⭐ Future Enhancements

- Add Random Forest & SVM
- Auto EDA (plots, correlations)
- Downloadable PDF report
- Pipeline export (YAML / JSON)
- Multi-model comparison dashboard

--- 

## 🙌 Author

Built with ❤️ by [Kai](https://github.com/kaix-404)

---

## 🛡️ License

MIT License — feel free to use and modify.
