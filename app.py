print("APP FILE STARTED")

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ✅ Correct path handling for Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")

# Load model
model = pickle.load(open(MODEL_PATH, "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = [
        float(request.form["N"]),
        float(request.form["P"]),
        float(request.form["K"]),
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["ph"]),
        float(request.form["rainfall"])
    ]

    prediction = model.predict([data])[0]

    return render_template("index.html", result=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
