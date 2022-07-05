import json
import logging
import os

import pandas as pd
from sklearn.metrics import classification_report

from diagnostics import model_predictions

FORMAT = "%(asctime)-15s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

with open("config.json", "r") as f:
    config = json.load(f)

OUTPUT_FOLDER_PATH = os.path.join(config["output_folder_path"])
TEST_DATA_PATH = os.path.join(config["test_data_path"], "testdata.csv")
OUTPUT_MODEL_PATH = os.path.join(config["output_model_path"], "report.json")
MODEL_VERSION_PATH = OUTPUT_MODEL_PATH + "/VERSION"


def create_report():
    """Calculate a confusion matrix using the test data and the deployed model.

    Args:
        None.

    Returns:
        None.
    """
    testdata = pd.read_csv(TEST_DATA_PATH)
    test_labels = testdata["exited"]
    preds_labels = model_predictions(testdata)["predictions"]
    report = classification_report(test_labels, preds_labels, output_dict=True)
    logging.info("Create report: %s", OUTPUT_MODEL_PATH)
    with open(OUTPUT_MODEL_PATH, "w") as json_file:
        json.dump(report, json_file)


if __name__ == "__main__":
    create_report()
