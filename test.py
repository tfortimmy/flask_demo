import requests
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

X = df.to_dict()

sample = df.loc[[0], :].to_dict()

df['labels'] = [iris['target_names'][t] for t in iris['target']]
y = df[['labels']].to_dict()

train_data = {
    'model_type': 'LogisticRegression',
    'X': X,
    'y': y
}

response = requests.post('http://127.0.0.1:5000/train', json=train_data)

print('Train Reponse Code:', response.status_code)

if response.status_code == 200:
    print(response.json())

    model_id = response.json()['model_id']

data = {
    'model_id': model_id,
    'data': sample
}

response = requests.post('http://127.0.0.1:5000/predict', json=data)

print(response.status_code)

if response.status_code == 200:
    print(response.json())
