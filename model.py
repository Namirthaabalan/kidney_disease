import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Update the file path to the actual location of your dataset
file_path = 'D:/BDA MICRO PROJECT_NEW/c1.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Check for missing values and fill them
data.fillna(data.mean(), inplace=True)

# Features and target variable
X = data.drop('Chronic Kidney Disease: yes', axis=1)
y = data['Chronic Kidney Disease: yes']

# Split the data into training and test sets (70% train, 30% test for better validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the Logistic Regression model with regularization (C=0.1)
model = LogisticRegression(C=0.1)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save the trained model as a pickle file
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler as a pickle file (important for scaling test data)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
