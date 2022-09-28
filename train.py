#! /usr/bin/env python3


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from ranking_utils.model import TrainingMode


@hydra.main(config_path="config", config_name="training")
def main(config: DictConfig) -> None:
    seed_everything(config.random_seed, workers=True)
    model = instantiate(config.ranker.model)
    data_module = instantiate(
        config.training_data, data_processor=instantiate(config.ranker.data_processor)
    )

    model.training_mode = data_module.training_mode = {
        "pointwise": TrainingMode.POINTWISE,
        "pairwise": TrainingMode.PAIRWISE,
        "contrastive": TrainingMode.CONTRASTIVE,
    }[config.training_mode]
    model.margin = config.margin
    model.in_batch_negatives = config.in_batch_negatives

    trainer = instantiate(config.trainer)
    trainer.fit(model=model, datamodule=data_module)
    if config.test:
        trainer.test(datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
