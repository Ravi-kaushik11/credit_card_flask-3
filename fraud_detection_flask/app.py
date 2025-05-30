from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import os

app = Flask(__name__)

# Get the absolute path to the models
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load models with error handling
try:
    with open(os.path.join(base_dir, 'fraud_detection_model.pkl'), 'rb') as file:
        rf_model = pickle.load(file)  # Random Forest

    with open(os.path.join(base_dir, 'xgboost_model.pkl'), 'rb') as file:
        xg_boost = pickle.load(file)  # XGBoost

    with open(os.path.join(base_dir, 'decision_tree_model.pkl'), 'rb') as file:
        dt_model = pickle.load(file)  # Decision Tree

    with open(os.path.join(base_dir, 'lightgbm_model.pkl'), 'rb') as file:
        lgbm_model = pickle.load(file)  # LightGBM
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    # You can set a flag here to show a warning in the UI if needed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate user fields
        user_name = request.form['User_Name']
        card_number = request.form['Card_Number']
        cvv = request.form['cvv']
        model_choice = request.form['model']

        if not re.fullmatch(r"[A-Za-z\s]+", user_name):
            return render_template('index.html', prediction_text="Invalid user name.")
        if not re.fullmatch(r"\d{12}", card_number):
            return render_template('index.html', prediction_text="Card number must be 12 digits.")
        if not re.fullmatch(r"\d{3}", cvv):
            return render_template('index.html', prediction_text="CVV must be 3 digits.")

        # Get model input features
        input_features = [float(x) for x in request.form.getlist('feature')]

        if len(input_features) != 30:
            return render_template('index.html', prediction_text="Expected 30 features, got " + str(len(input_features)))

        final_features = np.array([input_features])  # Shape: (1, 30)

        # Model prediction
        if model_choice == 'xgboost':
            prediction = xg_boost.predict(final_features)
        elif model_choice == 'random_forest':
            prediction = rf_model.predict(final_features)
        elif model_choice == 'decision_tree':
            prediction = dt_model.predict(final_features)
        elif model_choice == 'lightgbm':
            prediction = lgbm_model.predict(final_features)
        else:
            return render_template('index.html', prediction_text="Invalid model selected.")

        if prediction[0] == 1:
            result = "Fraud Detected"
            result_color = "text-red-600"
        else:
            result = "No Fraud"
            result_color = "text-green-600"
            
        return render_template('index.html', prediction_text=result, result_color=result_color)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", result_color="text-red-600")

if __name__ == "__main__":
    app.run(debug=True)
