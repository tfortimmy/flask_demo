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

        output = {}
        try:

            input_json = request.json

            model_id = input_json['model_id']

            model = joblib.load(f'{model_id}.pkl')

            df = pd.DataFrame(input_json['data'])

            predictions = model.predict(df)

            output['predictions'] = [int(p) for p in predictions]

        except Exception as e:
            output['error'] = str(e)

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
