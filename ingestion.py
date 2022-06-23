"""Module meant to ingest new data"""
import csv
import json
import logging
import os
from datetime import datetime

import pandas as pd

FORMAT = "%(asctime)-15s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

with open(file="config.json", mode="r", encoding="utf-8") as f:
    config = json.load(f)

SOURCE = config["input_folder_path"]
SINK = config["output_folder_path"]


def merge_multiple_dataframe():
    """Ingest a new data and delete duplicates.

    Args:
        None

    Returns:
        ingest_data (pandas.core.frame.DataFrame): Ingest data.
    """

    schema = {
        "corporation": pd.Series(dtype="str"),
        "lastmonth_activity": pd.Series(dtype="int"),
        "lastyear_activity": pd.Series(dtype="int"),
        "number_of_employees": pd.Series(dtype="int"),
        "exited": pd.Series(dtype="int"),
    }

    finaldata = pd.DataFrame(schema)

    source_path = os.path.join(os.getcwd(), SOURCE)
    source_list = os.listdir(source_path)

    for source_file in source_list:
        dataframe = pd.read_csv(os.path.join(source_path, source_file))
        finaldata = pd.concat([finaldata, dataframe])

        ingested_log = SINK + "/ingestedfiles.txt"
        ingesttime = str(datetime.now())
        row = [ingesttime, source_path, source_file, len(dataframe)]
        logging.info("Data Ingest: %s", row)
        with open(file=ingested_log, mode="a", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(row)

    finaldata.drop_duplicates(inplace=True)

    sink_path = os.path.join(SINK, "finaldata.csv")
    finaldata.to_csv(sink_path, index=False)


if __name__ == "__main__":
    merge_multiple_dataframe()
