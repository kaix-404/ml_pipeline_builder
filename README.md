# No-Code ML Pipeline Builder

A fully interactive no-code machine learning workflow builder built using Django + HTMX + Tailwind CSS.
Upload a dataset → preprocess → split → train → visualize → get results — all without writing a single line of Python.

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
│── builder/               # App
│── templates/
│   ├── base.html
│   ├── preprocess.html
│   ├── results.html
│   └── partials/
│── static/                # Tailwind + animations
│── final_model.pkl        # Saved model
│── manage.py
│── .gitignore
│── requirements.txt
```
