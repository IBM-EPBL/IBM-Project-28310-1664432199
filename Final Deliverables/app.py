import pickle
from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        # request all the input fields
        ph = float(request.form['ph value'])
        Hardness = float(request.form['Hardness'])
        Solids = float(request.form['Solids'])
        Chloramines = float(request.form['Chloramines'])
        Sulfate = float(request.form['Sulfate'])
        Conductivity = float(request.form['Conductivity'])
        Organic_carbon = float(request.form['Organic carbon'])
        Trihalomethanes = float(request.form['Trihalomethanes'])
        Turbidity = float(request.form['Turbidity'])

        # create numpy array for all the inputs
        val = np.array([ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity])

        # define save model and scaler path
        model_path = os.path.join('models', 'xgboost.sav')
        scaler_path = os.path.join('models', 'scaler.sav')

        # load the model and scaler
        model = pickle.load(open(model_path, 'rb'))
        scc = pickle.load(open(scaler_path, 'rb'))

        # transform the input data using pre fitted standard scaler
        data = scc.transform([val])

        # make a prediction for the given data
        res = model.predict(data)

        if res == 1:
            outcome = 'Potable'
        else:
            outcome = 'not potable'
        return render_template('index.html', result=outcome)
    return render_template('index.html')

# run application
if __name__ == "__main__":
    app.run(debug=True)