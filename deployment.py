"""Model Deployment"""
import json
import logging
import os

FORMAT = "%(asctime)-15s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

OUTPUT_FOLDER_PATH = config["output_folder_path"]
OUTPUT_MODEL_PATH = config["output_model_path"]
PROD_DEPLOYMENT_PATH = config["prod_deployment_path"]
MODEL_VERSION_PATH = OUTPUT_MODEL_PATH + "/VERSION"


def store_model_into_pickle():
    """
    Copy the latest model version, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory.

    Args:
        None

    Returns:
        None
    """

    with open(MODEL_VERSION_PATH, "r", encoding="utf-8") as textfile:
        _version = textfile.read()

    model_path = os.path.join(OUTPUT_MODEL_PATH, "model-" + _version + ".pkl")
    score_path = os.path.join(OUTPUT_MODEL_PATH, "latestscore.txt")
    ingest_path = os.path.join(OUTPUT_FOLDER_PATH, "ingestedfiles.txt")
    os.system(f"cp {model_path} {PROD_DEPLOYMENT_PATH}")
    os.system(f"cp {score_path} {PROD_DEPLOYMENT_PATH}")
    os.system(f"cp {ingest_path} {PROD_DEPLOYMENT_PATH}")
    os.system(f"cp {MODEL_VERSION_PATH} {PROD_DEPLOYMENT_PATH}")
    logging.info("Deploying model version: %s", _version)


if __name__ == "__main__":
    store_model_into_pickle()
