from flask import Flask, request, render_template, jsonify
from joblib import load

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """For rendering results on HTML GUI"""
    fixed_acidity = request.form["fixed_acidity"]
    volatile_acidity = request.form["volatile_acidity"]
    citric_acid = request.form["citric_acid"]
    residual_sugar = request.form["residual_sugar"]
    chlorides = request.form["chlorides"]
    free_sulfur_dioxide = request.form["free_sulfur_dioxide"]
    total_sulfur_dioxide = request.form["total_sulfur_dioxide"]
    density = request.form["density"]
    ph = request.form["ph"]
    sulphates = request.form["sulphates"]
    alcohol = request.form["alcohol"]

    prediction = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                 chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                                 ph, sulphates, alcohol]])[0]
    output = str()
    if prediction == 0:
        output = 'Bad'
    elif prediction == 1:
        output = 'Average'
    elif prediction == 2:
        output = 'Good'

    return render_template('index.html', prediction_text='{} wine quality'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """For direct API calls through request"""
    data = request.get_json(force=True)
    prediction_list = []
    for features in data:
        fixed_acidity = features["fixed_acidity"]
        volatile_acidity = features["volatile_acidity"]
        citric_acid = features["citric_acid"]
        residual_sugar = features["residual_sugar"]
        chlorides = features["chlorides"]
        free_sulfur_dioxide = features["free_sulfur_dioxide"]
        total_sulfur_dioxide = features["total_sulfur_dioxide"]
        density = features["density"]
        ph = features["pH"]
        sulphates = features["sulphates"]
        alcohol = features["alcohol"]
        prediction = model.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                     chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                                     ph, sulphates, alcohol]])[0]
        prediction_list.append(prediction)

    return jsonify({'prediction': str(prediction_list)})


if __name__ == '__main__':
    PATH = 'saved_models/'

    model = load(PATH + 'white_wine_quality_BaggingClassifier.joblib')
    print('Model loaded')

    model_columns = load(PATH + 'model_columns.joblib')
    print('Model columns loaded')

    app.run(host='0.0.0.0', port=5000, debug=True)
