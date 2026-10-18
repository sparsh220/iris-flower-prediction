
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_random_forest.pkl')


# 1. Clean the input data
def get_clean_data(form_data):
    sepal_length = float(form_data['sepal_length'])
    sepal_width = float(form_data['sepal_width'])
    petal_length = float(form_data['petal_length'])
    petal_width = float(form_data['petal_width'])

    cleaned_data = {
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    }

    return cleaned_data


# 2. Display the homepage
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# 3. Predict the flower species
@app.route("/predict", methods=["POST"])
def get_prediction():
    iris_data = request.form

    data_clean = get_clean_data(iris_data)

    iris_data_df = pd.DataFrame(data_clean)

    prediction = model.predict(iris_data_df)

    return render_template(
        "result.html",
        prediction=str(prediction[0])
    )


if __name__ == '__main__':
    app.run(debug=True)
