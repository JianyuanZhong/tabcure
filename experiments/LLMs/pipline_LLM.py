import argparse
import logging
import os

import pandas as pd
import torch
import wandb
import yaml
from peft import LoraConfig, PeftConfig, PeftModel
from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
from transformers import GenerationConfig, TrainingArguments

from tabcure import TabCure, seed_everything, set_logging_level
from tabcure.tabular_metrices.efficacy import (
    BinaryAdaBoostClassifier,
    BinaryDecisionTreeClassifier,
    BinaryLogisticRegression,
    BinaryMLPClassifier,
)

# Create a parser object
parser = argparse.ArgumentParser(description="Parse a config file")
# Add an argument for the config file name
parser.add_argument("--config", required=True, type=str, help="The path to a config file")
# Parse the command line arguments
args = parser.parse_args()


logger = set_logging_level(logging.INFO)
seed_everything(3407)

CURRENT_DIR = os.path.abspath(__file__)
CURRENT_DIR = "/".join(CURRENT_DIR.split("/")[:-1])

with open(os.path.join(CURRENT_DIR, args.config), "r") as f:
    config = yaml.safe_load(f)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = f"TabCure-{config['dataset']}"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


def train(config, real_data):
    # Initialize training configs
    trainer_config = TrainingArguments(**config["trainer"])
    peft_config = LoraConfig(**config["peft"])

    tabcure = TabCure(config["LLM"], trainer_config=trainer_config, peft_config=peft_config)
    try:
        tabcure.lora_fit(real_data, resume_from_checkpoint=True)
    except Exception:
        tabcure.lora_fit(real_data, resume_from_checkpoint=False)

    tabcure.model.save_pretrained(f"exp-{config['dataset']}/pretrained")

    del tabcure
    torch.cuda.empty_cache()


def generate(config, real_data, n_samples):
    peft_model_id = f"exp-{config['dataset']}/pretrained"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    tabcure = TabCure(peft_config.base_model_name_or_path)
    tabcure.model = PeftModel.from_pretrained(tabcure.model, peft_model_id)

    # update colomns info for newly loaded model
    tabcure._update_column_information(real_data)

    # tabcure = GReaT
    samples = tabcure.sample(n_samples, GenerationConfig(**config["generation"]), k=100)
    samples.to_csv(f"exp-{config['dataset']}/{config['dataset']}_synthetic.csv")
    return samples


def evaluate(real_test, synthetic_data, metadata):
    # run standard quanlity report
    evaluate_quality(real_data=real_test, synthetic_data=synthetic_data, metadata=metadata)

    diagnostic_report = run_diagnostic(real_data=real_test, synthetic_data=synthetic_data, metadata=metadata)
    diagnostic_result = diagnostic_report.get_results()
    logger.info(diagnostic_result)
    logger.info(diagnostic_report.get_properties())
    logger.info(diagnostic_report.get_details(property_name="Coverage"))
    logger.info(diagnostic_report.get_details(property_name="Synthesis"))
    logger.info(diagnostic_report.get_details(property_name="Boundaries"))

    # run MLE
    ada_score = BinaryAdaBoostClassifier.compute(
        test_data=real_test, train_data=synthetic_data, target="label", metadata=metadata
    )
    logger.info(f"MLE score for Ada: {ada_score}")
    lr_score = BinaryLogisticRegression.compute(
        test_data=real_test, train_data=synthetic_data, target="label", metadata=metadata
    )
    logger.info(f"MLE score for LR: {lr_score}")
    tl_score = BinaryDecisionTreeClassifier.compute(
        test_data=real_test, train_data=synthetic_data, target="label", metadata=metadata
    )
    logger.info(f"MLE score for TL: {tl_score}")
    mlp_score = BinaryMLPClassifier.compute(
        test_data=real_test,
        train_data=synthetic_data,
        target="label",
        metadata=metadata,
    )
    logger.info(f"MLE score for MLP: {mlp_score}")


def get_data(config):
    dataset_name = config["dataset"]
    try:
        logger.info("try to get data from sdv")
        real_data, metadata = download_demo(modality="single_table", dataset_name=dataset_name)
    except Exception:
        logger.info("load data from local instead")
        data_path = "../../data"
        real_data = pd.read_csv(os.path.join(data_path, f"{config['dataset']}.csv"))
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=real_data)

    real_data, real_test = train_test_split(real_data, test_size=0.2)
    return real_data, real_test, metadata


def main():
    dataset_name = config["dataset"]
    real_data, real_test, metadata = get_data(config)

    logger.info(f"training for dataset: {dataset_name}")
    train(config, real_data)

    logger.info(f"generating synthetic dataset: {dataset_name}")
    synthetic_data = generate(config, real_data, n_samples=100)

    logger.info(f"Evaluating synthetic dataset: {dataset_name}")
    evaluate(real_test, synthetic_data, metadata)


if __name__ == "__main__":
    main()
    wandb.finish()
