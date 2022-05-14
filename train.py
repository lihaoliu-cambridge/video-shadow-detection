import os
import yaml
import pytorch_lightning as pl
from typing import List
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

pl.seed_everything(1234)


def load_config(config_path: str, config_name: str) -> List[dict]:
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    dataset_config = config["dataset"]
    model_config = config["model"]
    trainer_config = config["train"]

    return dataset_config, model_config, trainer_config


def prepare_dataloader(dataset_config: dict, batch_size: int = 1) -> List[DataLoader]:
    if dataset_config["dataset_name"] == "ViSha_tvsd":
        from data_loader.visha_dataset_tvsd import ViSha_Dataset
    else:
        raise NotImplementedError("No dataset {}".format(dataset_config["dataset_name"]))
    
    dataset = ViSha_Dataset

    training_dataset = dataset("train", dataset_config)  # type: ignore
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size)

    testing_dataset = dataset("test", dataset_config)  # type: ignore
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

    return training_dataloader, testing_dataloader 


def prepare_model(model_config: dict) -> pl.LightningModule:
    if model_config["model_name"] == "tvsd":
        from model.tvsd_net import LightningNetwork
    else:
        raise NotImplementedError("No model {}".format(model_config["model_name"]))

    model = LightningNetwork(model_config)
    
    return model


def prepare_pl_trainer(trainer_config: dict) -> pl.Trainer:
    # tensorboard logger
    tb_dir = os.path.join(trainer_config["output_dir"], trainer_config["tb_dirname"])
    tb_logger = pl.loggers.TensorBoardLogger(tb_dir, name=trainer_config["experiment_name"], default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(trainer_config["output_dir"], trainer_config["checkpoint"]["ckpt_dirname"], trainer_config["experiment_name"]),
        monitor=None, 
        every_n_epochs=1,
        # select top k models, -1 means save all models from the end of each epoch
        save_top_k=-1
    )

    pl_trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        # basic config: gpus, epochs, output_dir
        gpus=trainer_config["gpus"],
        max_epochs=trainer_config["max_epochs"],
        default_root_dir=trainer_config["output_dir"],
        # gpu accelerate
        accelerator=trainer_config["accelerator"],
        strategy=trainer_config["strategy"],
        # grad clip
        gradient_clip_val=trainer_config["gradient_clip_val"],
        gradient_clip_algorithm=trainer_config["gradient_clip_algorithm"],
        # apex
        precision=16,
        amp_backend="apex",
        amp_level='O1',
        log_every_n_steps=1,
    )

    return pl_trainer


def main():
    # args and config
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./config/")
    parser.add_argument('--config_name', type=str, default="visha_tvsd_config.yaml")
    args = parser.parse_args()

    dataset_config, model_config, trainer_config = load_config(args.config_path, args.config_name)

    # prepare dataloader
    training_dataloader, _ = prepare_dataloader(dataset_config, batch_size=trainer_config["batch_size"])

    # prepare model
    model = prepare_model(model_config)

    # prepare training pipeline
    pl_trainer = prepare_pl_trainer(trainer_config)

    # training
    pl_trainer.fit(model, training_dataloader)


if __name__ == '__main__':
    main()
