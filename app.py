from flask import Flask, render_template, request, jsonify
from flask_pymongo import PyMongo
import numpy as np
import pandas as pd
import pickle as pkl
from bson import ObjectId

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://admin:admin123@cluster0.yjdrfwz.mongodb.net/MedicalDB?retryWrites=true&w=majority"
mongo = PyMongo(app)

@app.route("/")
def home():
    return jsonify({"Message":"Welcome New Brother"})

with open("model1.pkl","rb") as f:
    model = pkl.load(f)

SYMPTOMS = [
    "fever", "cough", "headache", "stomach_pain", "cold", "flu",
    "back_pain", "fatigue", "chest_pain", "shortness_of_breath",
    "skin_rash", "joint_pain", "sore_throat", "nausea", "dizziness",
    "vomiting", "diarrhea", "constipation", "loss_of_appetite",
    "high_heart_rate", "low_heart_rate", "high_blood_pressure",
    "low_blood_pressure", "sweating", "chills", "dehydration",
    "weakness", "anxiety", "depression", "insomnia", "ear_pain",
    "eye_pain", "blurred_vision", "runny_nose", "nose_bleed",
    "difficulty_swallowing", "abdominal_bloating", "burning_chest",
    "urination_pain", "frequent_urination", "blood_in_urine",
    "swollen_legs", "swollen_hands", "itching", "allergic_reaction",
    "body_ache", "neck_pain", "shoulder_pain", "loss_of_smell",
    "loss_of_taste"
]

def symptoms_to_vector(symptom_list):
    row = {s:0 for s in SYMPTOMS}   # Makin all features as 0
    for s in symptom_list:
        s = s.lower().strip()
        if s in row:
            row[s]=1        # Makin those feature as 1 which are entered by Patient..
    return pd.DataFrame([row])

@app.route("/predict/<patient_id>", methods = ["GET"])
def predict(patient_id):
    record = mongo.db.MediData.find_one({"_id":ObjectId(patient_id)})

    if not record:
        return jsonify({"error":"Patient ID not found"})
    
    symptom_list = record.get("data",[])

    x = symptoms_to_vector(symptom_list)

    prediction = model.predict(x)[0]

    urgency_labels = {
        0: "Low Urgency",
        1: "Medium Urgency",
        2: "High Urgency"
    }

    return render_template("index.html", response = urgency_labels[int(prediction)])




if __name__ == "__main__":

    app.run(debug=True)
