# ranking-models
This repository houses the implementations of our ranking models as well as some baselines. It supports __training__, __validation__, __testing__ and __re-ranking__. The following models are currently implemented:
* __Select & Rank__ [[1]](https://arxiv.org/abs/2106.12460)
* __BERT-DMN__ [[2]](https://arxiv.org/abs/2106.07316)
* __BERT__ [[3]](https://dl.acm.org/doi/10.5555/3295222.3295349) [[4]](https://arxiv.org/abs/1901.04085)
* __QA-LSTM__ [[5]](https://aclanthology.org/P16-1044/)
* __DMN__ [[6]](http://proceedings.mlr.press/v48/kumar16.pdf) [[7]](http://proceedings.mlr.press/v48/xiong16.pdf)

## Requirements
Install the packages from `requirements.txt`. Note that [ranking_utils](https://github.com/mrjleo/ranking-utils) has additional dependencies.

## Usage
We use [Hydra](https://hydra.cc/) for the configuration of all scripts, such as models, hyperparameters, paths and so on. Please refer to the documentation for instructions how to use Hydra.

### Pre-Processing
Currently, datasets must be pre-processed in order to use them for training. Refer to [this guide](https://github.com/mrjleo/ranking-utils#dataset-pre-processing) for a list of supported datasets and pre-processing instructions.

### Training and Evaluation
We use [PyTorch Lightning](https://pytorchlightning.ai/) for training and evaluation. Use the training script to train a new model and save checkpoints. At least the following options must be set: `ranker`, `training_data.data_dir` and `training_data.fold_name`. For a list of available rankers, run:
```
python train.py
```
For example, in order to train an __S&R-Linear__ ranker, run the following:
```
python train.py \
    ranker=select_and_rank_linear \
    training_data.data_dir=/path/to/preprocessed/files \
    training_data.fold_name=fold_0
```
The default configuration for training can be found in `config/training.yaml`. All defaults can be overriden via the command line.

You can further override or add new arguments to other components such as the `pytorch_lightning.Trainer`. Some examples are: 
* `+trainer.gpus=[0,1]` to train on GPUs `0` and `1`
* `+trainer.val_check_interval=5000` to validate every 5000 batches
* `+trainer.limit_val_batches=1000` to use only 1000 batches of the validation data

#### Training Mode
Currently, `pointwise` (cross-entropy), `pairwise` (max-margin) and `contrastive` loss functions are supported. Select a training loss by using
* `training_mode=pointwise`,
* `training_mode=pairwise` or
* `training_mode=contrastive`.

#### Model Parameters
Model (hyper)parameters can be overidden like any other argument. They are prefixed by `ranker.model` and `ranker.model.hparams`, respectively. Check each ranker's config (in `config/ranker`) for available options.

#### Output Files
The default behavior of Hydra is to create a new directory, `outputs`, in the current working directory. In order to use a custom output directory, override the `hydra.run.dir` argument.

#### Testing
By default, the trained model will be evaluated on the test set after the training finishes. This can be disabled by setting `test=False`.

#### Example Command
A complete command to train a model could look like this:
```
python train.py \
    ranker=select_and_rank_linear \
    training_mode=pairwise \
    random_seed=123 \
    training_data.data_dir=/path/to/preprocessed/files \
    training_data.fold_name=fold_0 \
    training_data.batch_size=32 \
    +trainer.gpus=[0,1] \
    hydra.run.dir=/path/to/output/files
```

### Prediction
You can also use a trained model to re-rank any existing test set or TREC runfiles:
```
python predict.py
```
The the default arguments can be found in `config/prediction.yaml`. Make sure to configure the `ranker` (i.e. hyperparameters) to be identical to the trained model. Futher, you must provide a checkpoint (`ckpt_path`) and data source (`prediction_data`).

#### Re-Ranking a Test Set
In order to re-rank a test set (as created by the data pre-processing script), set `prediction_data=h5` and provide `prediction_data.data_file` and `prediction_data.pred_file_h5`. For example:
```
python predict.py \
    ranker=select_and_rank_linear \
    ckpt_path=/path/to/checkpoint.ckpt \
    prediction_data=h5 \
    prediction_data.data_file=/path/to/data.h5 \
    prediction_data.pred_file_h5=/path/to/test.h5 \
    +trainer.gpus=[0,1] \
    hydra.run.dir=/path/to/output/files
```

#### Re-Ranking a TREC Runfile
Alternatively, a TREC runfile can be used to obtain the query-document pairs for re-ranking. Set `prediction_data=h5_trec` and provide `prediction_data.data_file` and `prediction_data.pred_file_trec`. For example:
```
python predict.py \
    ranker=select_and_rank_linear \
    ckpt_path=/path/to/checkpoint.ckpt \
    prediction_data=h5_trec \
    prediction_data.data_file=/path/to/data.h5 \
    prediction_data.pred_file_trec=/path/to/runfile.tsv \
    +trainer.gpus=[0,1] \
    hydra.run.dir=/path/to/output/files
```
