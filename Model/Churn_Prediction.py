# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Assuming the dataset is in a file named 'Telco-Customer-Churn.csv'
df = pd.read_csv('Model\Telco-Customer-Churn.csv')

# Step 2: Data Preprocessing

# Dropping customerID column (since it's not useful for the model)
df = df.drop(columns=['customerID'])

# Handling missing values
# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode categorical variables
# Binary variables (Yes/No)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

# Multi-class categorical variables
multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

# Encode gender (Male/Female)
df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Step 3: Feature Scaling for numerical columns
scaler = StandardScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 4: Split the dataset into training and test sets
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training with Random Forest
rf_model = LogisticRegression()
rf_model.fit(X_train, y_train)

# Step 6: Model Prediction
y_pred = rf_model.predict(X_test)

# Step 7: Model Evaluation

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# ROC Curve and AUC
y_pred_proba = rf_model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# print(f'ROC AUC Score: {roc_auc:.4f}')

def predict_churn(user_input):
    """
    Menerima input user berupa dictionary sesuai dengan fitur dataset,
    memproses input, dan memprediksi apakah pelanggan akan churn.

    Parameters:
    user_input (dict): Dictionary berisi nilai fitur pelanggan.

    Returns:
    str: Prediksi hasil ("Churn" atau "Tidak Churn").
    """
    # Masukkan input user ke DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Langkah preprocessing sama seperti dataset asli:
    # Encode gender
    user_df['gender'] = user_df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    
    # Binary encode
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        user_df[col] = user_df[col].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # One-hot encoding untuk fitur multi-class
    multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    user_df = pd.get_dummies(user_df, columns=multi_class_cols, drop_first=True)
    
    # Isi kolom dummy yang hilang dengan 0
    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[X.columns]  # Urutkan kolom sesuai X
    
    # Scale fitur numerik
    user_df[numerical_cols] = scaler.transform(user_df[numerical_cols])
    
    # Prediksi
    prediction = rf_model.predict(user_df)[0]
    return "Churn" if prediction == 1 else "Not Churn"


# Input user (sesuaikan nilai dengan fitur dataset Anda)
user_input = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 49,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Mailed check',
    'MonthlyCharges': 103.7,
    'TotalCharges': 5036.3
}

# Prediksi hasil
result = predict_churn(user_input)
print(f"Hasil Prediksi: {result}")
