from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from custom_transform import OutlierCapper

app = Flask(__name__)

try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Model file not found. Please ensure 'heart_disease_model.pkl' exists.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
    
        data = request.get_json()
        values = data['values']
        
        columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 
                   'resting ecg', 'max heart rate', 'exercise angina', 'oldpeak', 'ST slope']
        
        input_data = pd.DataFrame(np.array(values).reshape(1, -1), columns=columns)

        prediction = model.predict(input_data)
        
        result = "Heart Disease Detected (1)" if prediction[0] == 1 else "Normal (0)"
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)