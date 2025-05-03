import datetime as dt
import glob
import logging
import os
from copy import copy

import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CometLogger, CSVLogger, TensorBoardLogger

import settings


def parse_args():
    """Parse command line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--config",
        type=str,
        default=settings.CONFIG_PATH,
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    return args


def process_yaml_anchors_recursively(config: dict, value: any) -> any:
    """Recursively process YAML anchors in the configuration.
        Anchors are denoted by {{key}} and are replaced with the corresponding value from the config.

        Example:
            data:
                var: 123

            anchors:
                var_anchor1: {{data.var}}
                var_anchor2: {{data.var}}

    Args:
        config (dict): Configuration dictionary.
        value (any): Value to process.

    Raises:
        KeyError: If the key is not found in the configuration.

    Returns:
        any: Processed value.
    """
    if isinstance(value, str):
        if value.startswith("{{") and value.endswith("}}"):
            keys = value[2:-2].strip().split(".")
            t = copy(config)
            for key in keys:
                try:
                    t = t[key]
                except KeyError:
                    raise KeyError(
                        f"Key '{key}' not found in the configuration. Check the path: {value}"
                    )
            return t
        else:
            return value
    elif isinstance(value, dict):
        return {
            k: process_yaml_anchors_recursively(config, v) for k, v in value.items()
        }
    elif isinstance(value, list):
        return [process_yaml_anchors_recursively(config, v) for v in value]
    else:
        return value


def load_config(config_path: str):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Evaluate the string values in the config
        config = process_yaml_anchors_recursively(config, config)
    else:
        raise ValueError("Unsupported configuration file format. Use .yaml or .yml.")

    return config


def generate_run_name(config: dict) -> str:
    """Generate a unique run name based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        str: Generated run name.
    """
    run_name = f"{config['model']['arch']}_{config['model']['encoder_name']}"
    run_name += f"_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    return run_name


def log_run_to_dir(dirpath: str, config: dict) -> None:
    """Log the run to a directory.

    Args:
        dirpath (str): Directory path to save the logs.
        config (dict): Configuration dictionary.

    Raises:
        FileExistsError: If the run's directory already exists.
    """

    if os.path.exists(dirpath):
        raise FileExistsError(
            f"Directory {dirpath} already exists. Please choose a different name."
        )

    # Create run's directory
    os.makedirs(dirpath)

    # Save the configuration to a yaml file
    with open(os.path.join(dirpath, "config.yaml"), "w") as file:
        yaml.dump(config, file)


def get_console_logger(logger_name: str) -> logging.Logger:
    """Initialize a console logger.

    Args:
        logger_name (str): Logger name.

    Returns:
        logging.Logger: Initialized console logger.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(logger_name)
    return logger


def get_loggers(logger_configs: dict, run_name: str) -> list | bool | None:
    """Initialize the loggers based on the provided configuration.

    Args:
        logger_configs (dict): Configuration for the loggers.
        run_name (str): Experiment run name.

    Raises:
        NotImplementedError: Raised if logger not implemented.

    Returns:
        list | None: Initialized loggers.
    """
    if not logger_configs:
        return None

    loggers = []
    for logger_config in logger_configs:
        logger_name = logger_config["logger_name"]
        logger_params = logger_config["logger_params"]

        if logger_name == "comet":

            assert settings.COMET_API_KEY, "Comet API key is required for CometLogger."

            logger = CometLogger(
                api_key=settings.COMET_API_KEY, name=run_name, **logger_params
            )
            loggers.append(logger)
        elif logger_name == "csv":
            logger = CSVLogger(
                save_dir=settings.TRAIN_LOG_DIR, name=run_name, **logger_params
            )
            loggers.append(logger)
        elif logger_name == "tensorboard":
            logger = TensorBoardLogger(
                save_dir=settings.TRAIN_LOG_DIR, name=run_name, **logger_params
            )
            loggers.append(logger)
        else:
            raise NotImplementedError(f"Unsupported logger type: {logger_name}")

    return loggers


def finish_comet_run(logger: CometLogger, best_ckpt_path: str = None) -> None:
    """Finish Comet experiment run.

    Args:
        logger (CometLogger): Comet logger.
        dirname (str): Path to experiment run directory.
    """
    if best_ckpt_path:
        logger.experiment.log_model("model", best_ckpt_path)

    logger.experiment.end()


def get_callback(
    callback_name: str, callback_params: dict
) -> ModelCheckpoint | EarlyStopping | LearningRateMonitor:
    """Initialize the callback based on the provided configuration.

    Args:
        callback_name (str): Name of the callback to initialize.
        callback_params (dict): Configuration for the callback.

    Returns:
        ModelCheckpoint | EarlyStopping | LearningRateMonitor: Initialized callback.
    """
    if callback_name == "ModelCheckpoint":
        return ModelCheckpoint(**callback_params)
    elif callback_name == "EarlyStopping":
        return EarlyStopping(**callback_params)
    elif callback_name == "LearningRateMonitor":
        return LearningRateMonitor(**callback_params)
    else:
        raise ValueError(f"Unsupported callback type: {callback_name}")


def export_model_to_onnx(model, input_tensor, export_path):
    """Export the model to ONNX format.

    Args:
        model: The model to export.
        input_tensor: The input tensor for the model.
        export_path: The path to save the exported ONNX model.
    """
    import torch

    with torch.autocast("cuda", dtype=torch.float16):
        torch.onnx.export(
            model,
            input_tensor,
            export_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )


if __name__ == "__main__":
    from pprint import pprint

    config_path = "configs/config.yaml"
    config = load_config(config_path)

    pprint(config)
