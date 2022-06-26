"""Train model"""
import json
import logging
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

FORMAT = "%(asctime)-15s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

OUTPUT_FOLDER_PATH = os.path.join(config["output_folder_path"])
OUTPUT_MODEL_PATH = os.path.join(config["output_model_path"])


def train_model():
    """Train new model version"""

    traindata = pd.read_csv(OUTPUT_FOLDER_PATH + "/finaldata.csv")
    X = traindata[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y = traindata["exited"]

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    model.fit(X, y)

    with open(OUTPUT_MODEL_PATH + "/VERSION", "r", encoding="utf-8") as file:
        _version = int(file.read())
    _version += 1

    model_path = OUTPUT_MODEL_PATH + "/model-" + str(_version) + ".pkl"
    joblib.dump(model, model_path)

    logging.info("New version of the model has been trained: %s", _version)

    with open(OUTPUT_MODEL_PATH + "/VERSION", "w", encoding="utf-8") as file:
        file.write(str(_version))


if __name__ == "__main__":
    train_model()
