# Automated Model Selection and Hyperparameter Optimization Using Bayesian Optimization
Use Bayesian optimization techniques to automate model selection and hyperparameter tuning for machine learning models. Implement tools like Hyperopt or Optuna to explore the hyperparameter space efficiently and select the best-performing models based on cross-validated performance metrics. This approach enhances model accuracy and optimizes computational resources.

This project focuses on predicting whether a person has diabetes or not using the **Pima Indians Diabetes Dataset**. The main objective is to build a machine learning classification model using **Random Forest Classifier**, and improve its performance using various hyperparameter tuning techniques.

---

## 📂 Dataset

The dataset used is `diabetes.csv`, which contains diagnostic measurements for female patients and a binary target variable `Outcome` indicating the presence or absence of diabetes.

### Features
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- **Outcome** (Target variable)

📌 **Note**: Missing or zero values for features like `Glucose` were handled by replacing them with median values.

---

## 📌 Project Steps

### 1. **Data Preprocessing**
- Replaced invalid 0 values in 'Glucose' column with median.
- Defined `X` (features) and `y` (target).
- Performed train-test split (80-20).

---

### 2. **Model Training - Random Forest**
- Trained a baseline **RandomForestClassifier**.
- Evaluated with `confusion_matrix`, `accuracy_score`, and `classification_report`.

---

### 3. **Manual Hyperparameter Tuning**

Used customized parameters like:

```python
RandomForestClassifier(
    n_estimators=300, 
    criterion='entropy',
    max_features='sqrt',
    min_samples_leaf=10,
    random_state=100
)
```

---

### 🔍 4. **Randomized Search CV**

Explored a wide parameter space using `RandomizedSearchCV` to find the best hyperparameters:

* `n_estimators`: 200 - 2000  
* `max_features`: `'auto'`, `'sqrt'`, `'log2'`  
* `max_depth`: 10 - 1000  
* `min_samples_split`: [2, 5, 10, 14]  
* `min_samples_leaf`: [1, 2, 4, 6, 8]  
* `criterion`: `'entropy'`, `'gini'`

---

### 🔧 5. **Grid Search CV**

Refined the best parameters obtained from `RandomizedSearchCV` using `GridSearchCV` with a narrower range for more accuracy.

---

### 🤖 6. **Bayesian Optimization using Hyperopt**

Used **Hyperopt** to optimize hyperparameters using the **TPE (Tree-structured Parzen Estimator)** approach.

---

### 🧬 7. **Genetic Programming using TPOT**

Automated ML pipeline search using **TPOT**:

```python
TPOTClassifier(
    generations=5,
    population_size=24,
    config_dict={...},
    cv=4
)
```

---

### ⚙️ 8. **Optuna Optimization**

Used **Optuna**, an efficient hyperparameter optimization framework, to search for best-performing models (both **Random Forest** and **SVC**).

---

## ✅ Final Model Evaluation

Best-performing model after hyperparameter tuning was evaluated on the test set using:

* Confusion Matrix  
* Accuracy Score  
* Classification Report

---

## 📊 Example Metrics

* **Accuracy Score**: 0.77+  
* **Precision, Recall, F1-Score**: Evaluated for both classes (0 and 1)

---

## 🔧 Tools & Libraries Used

* Python (**Pandas**, **NumPy**)  
* **scikit-learn**  
* **matplotlib**, **seaborn** (for optional visualization)  
* **Hyperopt**  
* **TPOT**  
* **Optuna**

---

## 🚀 How to Run

**Clone the repository:**

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn hyperopt tpot optuna
```

**Run the notebook or script:**

```bash
python diabetes_prediction.py
```

---

## 📈 Future Work

* Add visualizations for feature importance  
* Try other models like **XGBoost**, **LightGBM**  
* Integrate with a simple web UI using **Flask** or **Streamlit** for real-time predictions

---

## 🙌 Acknowledgments

* Dataset from the **UCI Machine Learning Repository**

---


