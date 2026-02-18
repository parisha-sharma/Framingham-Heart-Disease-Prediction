# 🫀 Framingham Heart Disease Prediction

<div align="center">

![Heart Disease](https://img.shields.io/badge/Heart%20Disease-Prediction-8b5cf6?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-a855f7?style=for-the-badge&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-c084fc?style=for-the-badge&logo=scikit-learn&logoColor=white)

### *Predicting 10-year Coronary Heart Disease risk using machine learning*

[Overview](#-overview) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance) • [Contributing](#-contributing)

</div>

---

## 📊 Overview

This project implements a **machine learning classification pipeline** to predict the risk of developing Coronary Heart Disease (CHD) within 10 years, based on the Framingham Heart Study dataset. Using Logistic Regression and comprehensive data preprocessing, the model classifies patients as either **healthy (0)** or **at risk (1)** based on their health and lifestyle factors.

The dataset contains **4,238 patient records** with 16 features including age, smoking habits, blood pressure, cholesterol, BMI, glucose levels, and more.

### 🎯 Project Goals

- Perform comprehensive exploratory data analysis on patient health data
- Handle missing values and outliers with medically informed strategies
- Visualize relationships between health indicators and CHD risk
- Build a binary classification model using Logistic Regression
- Enable real-time CHD risk prediction through user input
- Understand key health factors influencing 10-year CHD risk

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔍 Data Analysis
- Dataset with 4,238 patient records and 16 features
- Comprehensive EDA with statistical summaries
- Outlier detection and handling via box plots
- Missing value imputation using median/zero strategies
- Correlation heatmap for feature relationship analysis
- Dropped low-correlation feature (education)

</td>
<td width="50%">

### 🤖 Machine Learning
- Binary classification (Healthy vs At Risk)
- Logistic Regression implementation
- Train-test split for model validation
- Accuracy Score: **85.49%**
- Interactive user input for real-time prediction
- Classification report with precision, recall, F1-score

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

<div align="center">

| Library | Purpose | Version |
|---------|---------|---------|
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation and analysis | Latest |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computations | Latest |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white) | Data visualization | Latest |
| ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white) | Statistical visualizations | Latest |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine learning algorithms | Latest |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive development | Latest |

</div>

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/Framingham-Heart-Disease-Prediction.git

# Navigate to project directory
cd Framingham-Heart-Disease-Prediction

# Install required dependencies
pip install -r requirements.txt
```

### Requirements File
```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🚀 Usage

### Running the Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Open the main notebook file
# Navigate to Framingham_Heart_Disease_Prediction.ipynb
```

### Quick Start Code

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('framingham.csv')

# Handle missing values
df['cigsPerDay'].fillna(df['cigsPerDay'].median(), inplace=True)
df['BPMeds'].fillna(0, inplace=True)
df['totChol'].fillna(df['totChol'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
df['heartRate'].fillna(df['heartRate'].median(), inplace=True)
df['glucose'].fillna(df['glucose'].median(), inplace=True)
df.drop('education', axis=1, inplace=True)

# Prepare features and target
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### 🩺 Predict for a New Patient

```python
# Input patient data for prediction
new_patient = [[1, 45, 0, 0, 1, 1, 0, 0, 198, 106, 71, 36, 90, 80]]
pred = model.predict(new_patient)
print(f"Predicted Disease (0=Healthy, 1=Possible): {pred[0]}")
```

---

## 📈 Model Performance

<div align="center">

### Logistic Regression Results

| Metric | Class 0 (Healthy) | Class 1 (At Risk) | Overall |
|--------|-------------------|-------------------|---------|
| **Precision** | 0.86 | 0.55 | 0.81 |
| **Recall** | 0.99 | 0.05 | 0.85 |
| **F1-Score** | 0.92 | 0.09 | 0.80 |
| **Support** | 724 | 124 | 848 |

### 🎯 Model Accuracy: **85.49%**

</div>

### Key Insights

- **Dataset Characteristics**: 4,238 patient records with 15 health features after preprocessing
- **Target Distribution**: Dataset is imbalanced — majority of patients are healthy (no CHD)
- **Top Risk Factors**: Age, systolic blood pressure, and glucose levels show strong correlation with CHD risk
- **Smoking**: Higher cigarettes per day is associated with elevated CHD risk
- **Missing Data Strategy**: Median imputation used for numeric health features with outliers (cigsPerDay, totChol, BMI, glucose); zero imputation for BPMeds

---

## 📁 Project Structure

```
Framingham-Heart-Disease-Prediction/
│
├── framingham.csv                              # Dataset
│
├── Framingham_Heart_Disease_Prediction.ipynb   # Main Jupyter notebook
│
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
└── LICENSE
```

---

## 🎨 Visualizations

The project includes comprehensive visualizations:

- 📊 **Correlation Heatmap** - Feature relationships across all 15 health variables
- 🩺 **Count Plot** - Total count of 10-Year CHD cases (healthy vs at risk)
- 👴 **KDE Plot** - Age distribution by CHD status
- 🚬 **Box Plot** - Cigarettes per day vs CHD risk
- ❤️ **Box Plot** - Systolic blood pressure vs CHD risk
- 📦 **Outlier Detection** - Box plots for cigsPerDay before and after median imputation

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository
2. 🔨 Create a new branch (`git checkout -b feature/improvement`)
3. 💾 Commit your changes (`git commit -am 'Add new feature'`)
4. 📤 Push to the branch (`git push origin feature/improvement`)
5. 🔃 Create a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Parisha Sharma**

- GitHub: [@parisha-sharma](https://github.com/parisha-sharma)
- LinkedIn: [parishasharma15](https://www.linkedin.com/in/parishasharma15)

---

## 🌟 Acknowledgments

- Dataset source: Kaggle (Framingham Heart Study)
- Inspired by real-world medical data science applications
- Built as part of learning journey in Data Science

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-8b5cf6?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data%20Science-Learning-a855f7?style=for-the-badge)

**Happy Learning! 🚀**

</div>
