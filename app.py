from flask import Flask, request, jsonify
import pandas as pd
import joblib

# start flask
app = Flask(__name__)

# when the post method detect, then redirect to success function


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
            model = joblib.load(f'models/{model_id}.pkl')

            # convert the data into a dataframe
            df = pd.DataFrame(input_json['data'])

            # TODO check the columns line up

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


# @app.route('/train', methods=['POST'])
# def train():
#     if request.method == 'POST':

#         output = {}

#         try:
#             input_json = request.json
#             df = pd.DataFrame(input_json)

#             predictions = model.predict(df)

#             output['predictions'] = predictions
#         except Exception as e:
#             output['error'] = e

#         return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
