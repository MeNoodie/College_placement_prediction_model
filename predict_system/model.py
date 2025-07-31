from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open(r"C:\Users\piyus\OneDrive\Documents\Desktop\PROJECT\predict_system\placement_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')  # This should be your input form page

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        stream = float(data['stream'])
        cgpa = float(data['cgpa'])
        internships = float(data['internships'])
        projects = float(data['projects'])

        features = np.array([[cgpa, stream, internships, projects]])
        probability = model.predict_proba(features)[0][1]
        percent = round(probability * 100, 2)

        return jsonify({'probability_percent': percent})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
