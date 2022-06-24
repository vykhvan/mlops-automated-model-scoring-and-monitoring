"""Model Scoring"""
import logging
import json
import os

import joblib
import pandas as pd
from sklearn import metrics

FORMAT = "%(asctime)-15s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

with open("config.json", "r") as f:
    config = json.load(f)

TEST_DATA = os.path.join(config["test_data_path"])
MODELS = os.path.join(config["output_model_path"])


def score_model():
    """This function should take a trained model,
    load test data, and calculate an F1 score
    for the model relative to the test data
    it should write the result to the latestscore.txt file.

    Args:
        None

    Returs:
        None
    """

    testdata = pd.read_csv(TEST_DATA + "/testdata.csv")

    with open(MODELS + "/VERSION", "r", encoding="utf-8") as file:
        _version = file.read()

    logging.info("Using a model version: %s", _version)

    with open(MODELS + "/model-" + _version + ".pkl", "rb") as file:
        model = joblib.load(file)

    X = testdata[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y = testdata["exited"]

    y_preds = model.predict(X)
    score = metrics.f1_score(y, y_preds)

    logging.info("Model scoring: %s", score)
    with open(MODELS + "/latestscore.txt", "w", encoding="utf-8") as file:
        file.write(str(score))


if __name__ == "__main__":
    score_model()
