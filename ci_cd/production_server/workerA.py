from celery import Celery

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json


model_json_file = './model.json'
model_weights_file = './model.weights.h5'
data_file = './github-repository-data.csv'

FEATURE_COLUMNS = [
    "forks_count",
    "subscribers_count",
    "open_issues_count",
    "size",
    "network_count",
    "has_wiki",
    "has_pages",
    "has_issues",
    "topics_count",
    "age_days",
    "language_encoded"
]

def load_data():
    df = pd.read_csv(data_file)
    df["language_encoded"] = pd.factorize(df["language"])[0]
    df["log_stars"] = np.log1p(df["stargazers_count"])
    X = df[FEATURE_COLUMNS].values
    y = df["log_stars"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def load_model():
    # load json and create model
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_file)
    #print("Loaded model from disk")
    return loaded_model

# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbit:5672/'
CELERY_RESULT_BACKEND = 'rpc://'
# Initialize Celery
celery = Celery('workerA', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

@celery.task()
def add_nums(a, b):
   return a + b

@celery.task
def get_predictions():
    results = {}
    X, y = load_data()
    loaded_model = load_model()
    log_predictions = loaded_model.predict(X).flatten()
    predictions = np.expm1(log_predictions)  # convert from log space back to star counts
    results['y'] = np.expm1(y).astype(int).tolist()
    results['predicted'] = predictions.tolist()
    return results

@celery.task
def get_accuracy():
    X, y = load_data()
    loaded_model = load_model()
    loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    score = loaded_model.evaluate(X, y, verbose=0)
    return score[1]  # MAE

