import json
import os

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
        None.
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


if __name__ == "__main__":
    print(dataframe_summary())
