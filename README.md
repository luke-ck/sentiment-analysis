# sentiment-analysis
Active Meta Learning with Transformer Ensembles for Sentiment Analysis

This repository contains the code for the project done as part of the CIL course at ETH Zurich. The project report can be found [here](Report.pdf). This version fixes the bug mentioned in the report and bumps Pytorch-Lightning and Pytorch to version 2.0. We scored 6th place in the [Kaggle](https://www.kaggle.com/competitions/cil-text-classification-2022) competition out of 25 teams (87 participants). 

## Setup
To install the required packages, run
```
pip install -r requirements.txt
```

## Data
The data is provided in the `/data` folder. For more information refer to the [README](data/README.md) in the data folder.

## Training
To train the model, first configure the 'config.yaml' file in the `/config` folder. Most importantly, make sure to configure the 
**mode** type. We currently support *training, testing, active_learning, meta_learning*. The *active_learning* and *meta_learning* modes should be employed only after 
training has been done, as they require a checkpoint of the model to be loaded. To speed up development, we cache the preprocessed datasets, so if you change some parameters make sure to delete them before retraining.
We also tightly integrate with [Weights & Biases](https://wandb.ai/site) for logging and visualization. 

After everything has been configured, run

```
python main.py
```

For more insights, consult the WandB report [here](https://wandb.ai/pmlr/sentiment-analysis/reports/Sentiment-Analysis--Vmlldzo0Mjg3MjU3).