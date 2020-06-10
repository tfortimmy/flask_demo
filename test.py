import requests
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

data = {
    'model_id': 'model',
    'data': df.loc[[0], :].to_dict()
}


# x = df.loc[[0], :].to_dict()

# print(x)


# # tmp = pd.DataFrame.from_dict(x)
# tmp = pd.DataFrame(x)

# print(tmp)

response = requests.post('http://127.0.0.1:5000/predict', json=data)

print(response.status_code)

if response.status_code == 200:
    print(response.json())
