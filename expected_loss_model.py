import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# Load the dataset
file_path = 'Task 3 and 4_Loan_Data.csv'  # Change this to your file path if needed
loan_data = pd.read_csv(file_path)

# Prepare features and target
X = loan_data.drop(columns=['customer_id', 'default'])
y = loan_data['default']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
auc_logreg = roc_auc_score(y_test, y_pred_logreg)

# Decision Tree model
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict_proba(X_test)[:, 1]
auc_tree = roc_auc_score(y_test, y_pred_tree)

# Recovery rate
RECOVERY_RATE = 0.10

# Function to calculate expected loss using both models
def expected_loss(loan_features):
    df = pd.DataFrame([loan_features])
    X_scaled = scaler.transform(df)
    pd_logreg = logreg.predict_proba(X_scaled)[:, 1][0]
    pd_tree = tree.predict_proba(df)[:, 1][0]
    loan_amount = loan_features['loan_amt_outstanding']
    el_logreg = pd_logreg * loan_amount * (1 - RECOVERY_RATE)
    el_tree = pd_tree * loan_amount * (1 - RECOVERY_RATE)
    return {
        'PD_LogisticRegression': round(pd_logreg, 4),
        'ExpectedLoss_LogisticRegression': round(el_logreg, 2),
        'PD_DecisionTree': round(pd_tree, 4),
        'ExpectedLoss_DecisionTree': round(el_tree, 2)
    }

# Example loan feature set
if __name__ == "__main__":
    example_features = {
        'credit_lines_outstanding': 3,
        'loan_amt_outstanding': 5000,
        'total_debt_outstanding': 15000,
        'income': 60000,
        'years_employed': 4,
        'fico_score': 650
    }
    result = expected_loss(example_features)
    print("Example Expected Loss:")
    print(result)
