from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("dbs_model.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        num = float(request.form.get("rates"))
        prediction = model.predict([[num]])[0][0]
        rounded_prediction = round(prediction, 2)  # Round to 2 decimal places
        return render_template("index.html", result=rounded_prediction)
    else:
        return render_template("index.html", result="Waiting...")


if __name__ == "__main__":
    app.run(debug=True)
