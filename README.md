# Flask Demo
Here I will go through a very simple example of how you can use flask to act as an endpoint for a ML application.

## Setup
For this project we require a Python version `>=3.6`
``` sh
python -m venv venv
source venv/bin/activate
pip install -r requirements
```

## Running
To run the application it is pretty simple.
``` sh
python app.py
```

## Requests
There are two forms of request that can be processed by the application.
### Train
To train a model you will need to provide:
* `model_type`
    This is the type of model you want to train. ['LogisticRegression', 'RandomForest', 'SVC', 'DecisionTree']
* `X`
    The data to train on. (From `pd.DataFrame.to_dict`)
* `y`
    The associated label. (From `pd.DataFrame.to_dict`)

This will need to be sent to the application as a json.

The app will return the associated `model_id` for the newly created model.

### Predict
To generate the prediction from a previously trained model you will need:
* `model_id`
    The id of the model to generate the prediction.
* `data`
    The data to predict. (From `pd.DataFrame.to_dict`)

This will need to be sent to the application as a json.

The application will then return the predictions and the probabilities for each of the data points sent.