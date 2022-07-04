import json
import os
import subprocess
import timeit

import joblib
import numpy as np
import pandas as pd

with open("config.json", "r") as f:
    config = json.load(f)

OUTPUT_FOLDER_PATH = os.path.join(config["output_folder_path"])
TEST_DATA_PATH = os.path.join(config["test_data_path"], "testdata.csv")
PROD_DEPLOYMENT_PATH = os.path.join(config["prod_deployment_path"])
MODEL_VERSION_PATH = PROD_DEPLOYMENT_PATH + "/VERSION"


def dataframe_summary():
    """Calculate summary statistics for training data.

    Args:
        None.

    Returns:
        summary_statistics: base statistics.
    """
    traindata = pd.read_csv(OUTPUT_FOLDER_PATH + "/finaldata.csv")
    columns = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    summary_statistics = {}
    for column in columns:
        mean = np.mean(traindata[column])
        median = np.median(traindata[column])
        std = np.std(traindata[column])
        summary_statistics.update(
            {column: {"mean": mean, "median": median, "std": std}}
        )
    return summary_statistics


def data_integrity_check():
    """Check data integrity for training data.

    Args:
        None.

    Returns:
        integiry: na values statistics.
    """
    traindata = pd.read_csv(OUTPUT_FOLDER_PATH + "/finaldata.csv")
    nas = list(traindata.isna().sum())
    napercents = [nas[i] / len(traindata.index) for i in range(len(nas))]
    integrity = {}
    for i in range(len(napercents)):
        integrity.update({traindata.columns[i]: napercents[i]})
    return integrity


def execution_time():
    """Calculate timing of training and ingestion.

    Args:
        None.

    Returns:
        exec_timing: timing of execution"""
    exec_timing = {}
    for step in ["ingestion.py", "training.py"]:
        starttime = timeit.default_timer()
        os.system(f"python {step}")
        timing = timeit.default_timer() - starttime
        exec_timing.update({step: timing})
    return exec_timing


def outdated_packages_list():
    """Return outdated packages.

    Args:
        None.

    Returns:
        installed: outdated packages list.

    """
    installed = subprocess.check_output(["pip", "list", "--outdated"])
    return installed.decode("utf-8")


def model_predictions(input_data):
    """Load last model version and create inference.

    Args:
        input_data: pandas.DataFrame with features.

    Returns:
        predictions: pandas.DataFrame with model inference.
    """

    with open(MODEL_VERSION_PATH, "r", encoding="utf-8") as textfile:
        _version = textfile.read()

    with open(PROD_DEPLOYMENT_PATH + "/model-" + _version + ".pkl", "rb") as file:
        model = joblib.load(file)

    input_data = input_data[
        ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    ]
    predictions = model.predict(input_data)

    return {"predictions": list(predictions)}


if __name__ == "__main__":
    print(dataframe_summary())
    print(data_integrity_check())
    print(execution_time())
    print(outdated_packages_list())
    print(model_predictions(pd.read_csv(TEST_DATA_PATH)))
