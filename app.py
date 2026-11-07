
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import date, timedelta

app = Flask(__name__)
CORS(app)

yield_model = joblib.load("yield_model.pkl")
harvest_model = joblib.load("harvest_model.pkl")

@app.route("/")
def home():
    return jsonify({
        "message": "HarvestSync AI API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        sowing_date = date.fromisoformat(data["sowing_date"])

        farmer = {
            "Crop": data["crop"],
            "Area": float(data["area"]),
            "Soil": data["soil"],
            "Rainfall": float(data["rainfall"]),
            "Temperature": float(data["temperature"]),
            "Irrigation": data["irrigation"],
            "Sowing_Day": sowing_date.timetuple().tm_yday
        }

        yield_input = pd.DataFrame([farmer])

        predicted_yield = yield_model.predict(yield_input)[0]

        harvest_features = [
            "Crop", "Soil", "Rainfall",
            "Temperature", "Irrigation", "Sowing_Day"
        ]

        days = harvest_model.predict(
            yield_input[harvest_features]
        )[0]

        harvest_date = sowing_date + timedelta(days=round(days))

        return jsonify({
            "crop": farmer["Crop"],
            "expected_yield_kg": round(float(predicted_yield)),
            "harvest_days": round(float(days)),
            "expected_harvest_date": harvest_date.isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
