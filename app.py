from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and train the model with your dataset
def train_model():
    # Load your dataset (adjust path as necessary)
    file_path = 'c1.csv'
    data = pd.read_csv(file_path)

    # Fill missing values (if any)
    data.fillna(data.mean(), inplace=True)

    # Features and target variable (update column names accordingly)
    X = data.drop('Chronic Kidney Disease: yes', axis=1)  # Assuming 'Chronic Kidney Disease: yes' is the target
    y = data['Chronic Kidney Disease: yes']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the logistic regression model
    model = LogisticRegression(C=0.5)
    model.fit(X_scaled, y)

    return model, scaler

# Train the model once and reuse
model, scaler = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get numeric inputs from the form (update based on the number of features in the dataset)
    try:
        input_features = [
            float(request.form['age']), 
            float(request.form['blood_pressure']),
            float(request.form['specific_gravity']),
            float(request.form['albumin']),
            float(request.form['sugar']),
            float(request.form['blood_glucose_random']),
            float(request.form['blood_urea']),
            float(request.form['serum_creatinine']),
            float(request.form['sodium']),
            float(request.form['potassium']),
            float(request.form['hemoglobin']),
            float(request.form['packed_cell_volume']),
            float(request.form['white_blood_cells']),
            float(request.form['red_blood_cells']),
            int(request.form['red_blood_cells_normal']),
            int(request.form['pus_cells_normal']),
            int(request.form['pus_cell_clumps_present']),
            int(request.form['bacteria_present']),
            int(request.form['hypertension']),
            int(request.form['diabetes_mellitus']),
            int(request.form['coronary_artery_disease']),
            int(request.form['appetite_poor']),
            int(request.form['pedal_edema']),
            int(request.form['anemia'])
        ]
    except ValueError:
        return "Invalid input. Please enter valid numbers."

    # Convert input into the format for prediction
    input_data = np.array(input_features).reshape(1, -1)

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]

    result = "Positive for CKD" if prediction == 1 else "Negative for CKD"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
