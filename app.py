from flask import Flask, request, jsonify
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# start flask
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        # import pdb
        # pdb.set_trace()

        # A dict for the output to be stored
        output = {}
        try:

            # the input data
            input_json = request.json

            # extract the model to use
            model_id = input_json['model_id']
            saved_model = joblib.load(f'models/{model_id}.pkl')

            model = saved_model['model']
            assert model_id == saved_model['model_id']
            columns = saved_model['columns']

            # convert the data into a dataframe
            df = pd.DataFrame(input_json['data'])

            # Assert the columns for the input data are the same as for the trained model
            assert np.all(columns == df.columns), "Columns must match"

            # get the predictions and scores
            predictions = model.predict(df)
            scores = model.predict_proba(df)

            # store the model outputs in the output dict
            # need to convert the outputs into something json compatible
            output['predictions'] = [int(p) for p in predictions]
            # take the max score (the one associated with the prediction)
            output['scores'] = [float(max(s)) for s in scores]

        except Exception as e:
            # If we experience an error then return it as part of the output
            output['error'] = str(e)

        # To return it in flask we need to convert it to json
        return jsonify(output)


@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':

        # To send to
        output = {}

        try:
            # extract the json
            input_json = request.json

            # what type of model are we training
            model_type = input_json['model_type']

            # extract the data
            X = pd.DataFrame(input_json['X'])
            y = pd.DataFrame(input_json['Y'])

            # allow the user to specify added arguments
            kwargs = input_json.get('kwargs', {})

            # build the model according to the model selection
            if model_type == 'LogisticRegression':
                model = LogisticRegression(**kwargs)
            elif model_type == 'RandomForest':
                model = RandomForestClassifier(**kwargs)
            elif model_type == 'DecisionTree':
                model = DecisionTreeClassifier(**kwargs)
            elif model_type == 'SVC':
                # as we want to return probabilities we need this param for the SVC
                model = SVC(probability=True, **kwargs)
            else:
                raise ValueError

            # fit the model
            model.fit(X, y)

            # the model id is the hexadecimal representation of the hash of the model
            model_id = hex(hash(model))

            # the columns used to train
            columns = X.columns

            # save all of the required elements in a dict
            to_save = {
                'model_id': model_id,
                'columns': columns,
                'model': model
            }

            # save the model on the server
            joblib.dump(to_save, f'models/{model_id}.pkl')

            # the model id so the user can call the model they trained
            output['model_id'] = model_id

        except Exception as e:
            # If we experience an error then return it as part of the output
            output['error'] = str(e)

        # convert the output to json
        return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
