from warnings import filterwarnings

import comet_ml  # type: ignore # To get all data logged automatically
import lightning as L
from lightning.pytorch.loggers import CometLogger

import settings
from data import SegmentationDataModule
from model import SegmentationModel
from utils import (
    finish_comet_run,
    generate_run_name,
    get_callback,
    get_console_logger,
    get_loggers,
    load_config,
    log_run_to_dir,
    parse_args,
)

if __name__ == "__main__":

    filterwarnings("ignore")
    console_logger = get_console_logger("TrainLogger")

    # --- Parse command line arguments ---
    args = parse_args()

    # --- Load Hyperparameters ---
    config = load_config(args.config)
    data_config = config["data"]
    training_config = config["trainer"]

    console_logger.info(f"Loaded configuration from {settings.CONFIG_PATH}")

    # --- Set up run name ---
    run_name = generate_run_name(training_config)
    console_logger.info(f"Generated run name: {run_name}")

    # --- Track experiment into local folder ---
    dirpath = f"{settings.LOG_DIR}/train/{run_name}"
    log_run_to_dir(dirpath, config)

    # --- Reproducibility ---
    L.seed_everything(
        seed=training_config["common"]["seed"], workers=True, verbose=False
    )
    console_logger.info(f"Set random seed to {training_config['common']['seed']}")

    # --- Initialize Data Module ---
    data_module = SegmentationDataModule(data_config=data_config)
    console_logger.info("Initialized DataModule")

    # --- Initialize Trainer module ---
    model = SegmentationModel(trainer_config=training_config)
    console_logger.info("Initialized Model")

    # --- Initialize Experiment Logger ---
    loggers = get_loggers(logger_configs=training_config["loggers"], run_name=run_name)
    console_logger.info("Initialized Experiment Loggers")

    # --- Initialize Callbacks ---
    # This callback is mandatory
    model_checkpoint = get_callback(
        "ModelCheckpoint",
        {"dirpath": dirpath, **training_config["callbacks"]["model_checkpoint"]},
    )

    callbacks = [
        # These callbacks are optional
        get_callback(
            callback_config["callback_name"], callback_config["callback_params"]
        )
        for callback_config in training_config["callbacks"]["optional_callbacks"]
    ]
    callbacks.append(model_checkpoint)
    console_logger.info("Initialized Callbacks")

    # --- Initialize Trainer ---
    trainer = L.Trainer(
        accelerator=training_config["common"]["accelerator"],
        devices=training_config["common"]["devices"],
        max_epochs=training_config["common"]["epochs"],
        logger=loggers,
        callbacks=callbacks,
        precision=training_config["common"]["precision"],
        log_every_n_steps=training_config["common"]["log_every_n_steps"],
        limit_train_batches=4 if data_config["dry_run"] else None,
        limit_val_batches=4 if data_config["dry_run"] else None,
        limit_test_batches=4 if data_config["dry_run"] else None,
    )
    console_logger.info("Initialized Trainer")

    # --- Train the model ---
    console_logger.info("Starting training...")
    trainer.fit(model, datamodule=data_module)
    console_logger.info("Training finished.")

    # --- Best model checkpoint ---
    best_model_path = model_checkpoint.best_model_path

    # --- Test the model ---
    console_logger.info("Starting testing...")
    trainer.test(model, datamodule=data_module, ckpt_path=best_model_path)
    console_logger.info("Testing finished.")

    # --- End Comet Experiment (if used) ---
    if isinstance(loggers, list):
        for logger in loggers:
            if isinstance(logger, CometLogger):
                finish_comet_run(logger, best_model_path)
                console_logger.debug("Comet experiment ended.")

    console_logger.info("Script finished.")
