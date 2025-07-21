from flask import Flask, render_template, request
import numpy as np
import pickle
import gzip

# Load model and scaler
with gzip.open('HeartFailiure_PredictionModel.pkl', 'rb') as f:
    model = pickle.load(f)


# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [
            float(request.form['age']),
            float(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            float(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            float(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            float(request.form['sex']),
            float(request.form['smoking']),
            float(request.form['time'])
        ]
        input_array = np.asarray(input_features).reshape(1, -1)
        prediction = model.predict(input_array)

        result = "⚠️ Patient is likely to have heart failure." if prediction[0] == 1 else "✅ Patient is unlikely to have heart failure."
       
        return render_template('index.html', result=result)

    except Exception as e:
        return f"Something went wrong: {e}"

if __name__ == '__main__':
    app.run(debug=True)
