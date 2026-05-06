import joblib
import os
import numpy as np

class MLService:
    def __init__(self):
        self.base_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        self.models = {}
        self.encoders = {}
        self.load_resources()

    def load_resources(self):
        try:
            self.models['diag'] = joblib.load(os.path.join(self.base_path, 'xgboost_diag.joblib'))
            self.models['cost'] = joblib.load(os.path.join(self.base_path, 'rf_cost_regressor.joblib'))
            self.models['scaler'] = joblib.load(os.path.join(self.base_path, 'scaler.joblib'))
            self.models['pca'] = joblib.load(os.path.join(self.base_path, 'pca_model.joblib'))
            self.models['kmeans'] = joblib.load(os.path.join(self.base_path, 'kmeans_model.joblib'))
            
            for col in ['gender', 'blood_type', 'medical_condition', 'admission_type', 'target']:
                self.encoders[col] = joblib.load(os.path.join(self.base_path, f'le_{col}.joblib'))
            print("✅ Modèles Synchronisés chargés.")
        except Exception as e:
            print(f"⚠️ Erreur chargement : {e}")

    def predict(self, data, task='diagnostic'):
        age = int(data.get('Age'))
        stay_duration = int(data.get('Stay_Duration', 1))
        health_risk = (age * stay_duration) / 100
        
        gender_code = self.encoders['gender'].transform([data.get('Gender')])[0]
        bt_code = self.encoders['blood_type'].transform([data.get('Blood_Type')])[0]
        cond_code = self.encoders['medical_condition'].transform([data.get('Medical_Condition')])[0]
        adm_code = self.encoders['admission_type'].transform([data.get('Admission_Type')])[0]

        features = [[age, gender_code, bt_code, cond_code, adm_code, stay_duration, health_risk]]
        
        if task == 'diagnostic':
            pred_code = self.models['diag'].predict(features)[0]
            prediction = self.encoders['target'].inverse_transform([pred_code])[0]
            probs = self.models['diag'].predict_proba(features)[0]
            return {"result": prediction, "confidence": round(float(max(probs)) * 100, 2), "type": "Diagnostic"}
        
        elif task == 'cost':
            cost = self.models['cost'].predict(features)[0]
            return {"result": f"{round(float(cost), 2)} $", "type": "Billing Estimation"}
            
        else: # Segmentation (Arwa)
            scaled = self.models['scaler'].transform(features)
            coords = self.models['pca'].transform(scaled)
            cluster = int(self.models['kmeans'].predict(coords)[0])
            
            # --- SYNCHRONISATION VISUELLE ---
            # On force le point à se déplacer sur l'axe X :
            # Cluster 0 (Low) -> Gauche (autour de -5)
            # Cluster 1 (Medium) -> Centre (autour de 0)
            # Cluster 2 (High) -> Droite (autour de 5)
            visual_x = (cluster - 1) * 5 + (coords[0][0] % 2) # On garde un peu de la variation réelle
            
            risk_map = {0: "Low Risk Profile", 1: "Medium Risk Profile", 2: "High Risk Profile"}
            return {
                "result": risk_map.get(cluster),
                "pca_x": float(visual_x),
                "pca_y": float(coords[0][1]),
                "cluster": cluster,
                "type": "Patient Segmentation"
            }

ml_service = MLService()
