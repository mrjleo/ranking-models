#! /usr/bin/env python3


from collections import defaultdict
from pathlib import Path

import hydra
import torchtext
from hydra.utils import instantiate
from omegaconf import DictConfig
from ranking_utils import write_trec_eval_file

torchtext.disable_torchtext_deprecation_warning()


@hydra.main(config_path="config", config_name="prediction", version_base="1.3")
def main(config: DictConfig) -> None:
    dataset = instantiate(
        config.prediction_data, data_processor=instantiate(config.ranker.data_processor)
    )
    trainer = instantiate(config.trainer)

    ids_iter = iter(dataset.ids())
    result = defaultdict(dict)
    for item in trainer.predict(
        model=instantiate(config.ranker.model),
        dataloaders=instantiate(
            config.data_loader, dataset=dataset, collate_fn=dataset.collate_fn
        ),
        ckpt_path=config.ckpt_path,
    ):
        for index, score in zip(
            item["indices"].detach().numpy(),
            item["scores"].detach().numpy(),
        ):
            i, q_id, doc_id = next(ids_iter)
            assert index == i
            result[q_id][doc_id] = score

    # include the rank in the file name, otherwise multiple processes compete with each other
    out_file = Path.cwd() / f"predictions_{trainer.global_rank}.tsv"
    write_trec_eval_file(out_file, result, config.name)


if __name__ == "__main__":
    main()
