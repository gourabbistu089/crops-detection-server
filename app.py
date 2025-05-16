from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
# CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})
CORS(app, resources={r"/predict": {"origins": "*"}})


# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['N'], data['P'], data['K'], data['temperature'],
                data['humidity'], data['ph'], data['rainfall']]
    prediction = model.predict([features])
    return jsonify({'crop': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
