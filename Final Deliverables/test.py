try:
    import pytest
    import unittest
    from app import app
    import pickle, os
    import numpy as np
    from unittest.mock import patch
    import requests
    import json

except Exception as e:
    print("Some modules are missing {}".format(e))


class FlaskTest(unittest.TestCase):

    # check for the response 200
    def test_route(self):
        tester = app.test_client(self)
        response = tester.get('/')
        statuscode = response.status_code
        self.assertEqual(statuscode, 200)

    # test for content type
    def test_content(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.content_type, 'text/html; charset=utf-8')

    # test for post request data
    @patch('requests.post')
    def test_post_request(self, mock_post):
        info = {'ph value': 1, 'Hardness': 1, 'Solids': 1, 'Chloramines': 1, 'Sulfate':1, 'Conductivity':1, 'Organic carbon':1, 'Trihalomethanes':1, 'Turbidity':1}
        resp = requests.post("/", data=json.dumps(info), headers={'Content-Type': 'application/json'})
        mock_post.assert_called_with("/", data=json.dumps(info),
                                     headers={'Content-Type': 'application/json'})

    # test for models
    def test_model(self):
        val = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        model_path = os.path.join('models', 'xgboost.sav')
        scaler_path = os.path.join('models', 'scaler.sav')
        model = pickle.load(open(model_path, 'rb'))
        scc = pickle.load(open(scaler_path, 'rb'))
        data = scc.transform([val])
        # make a prediction for the given data
        res = model.predict(data)
        self.assertEqual(np.array([0]), res)


if __name__ == "__main__":
    unittest.main()