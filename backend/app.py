import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from services.ml_service import ml_service

app = Flask(__name__)
# On autorise toutes les origines pour le déploiement
CORS(app)

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    return jsonify({
        "genders": ml_service.encoders['gender'].classes_.tolist(),
        "blood_types": ml_service.encoders['blood_type'].classes_.tolist(),
        "conditions": ml_service.encoders['medical_condition'].classes_.tolist(),
        "admission_types": ml_service.encoders['admission_type'].classes_.tolist()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    task = data.get('task', 'diagnostic')
    try:
        result = ml_service.predict(data, task)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Render utilise la variable d'environnement PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
