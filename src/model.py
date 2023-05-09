import math
from typing import Dict, List, Union, Tuple, Any
import pytorch_lightning as pl
import torch
from bunch import Bunch
from torch import nn
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR, _LRScheduler

from tokenizer import PreTrainedTokenizer
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModel
from data_loader import TransformerDataLoader
from torchmetrics import AUROC, Accuracy, ConfusionMatrix
from torch.optim.optimizer import Optimizer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TOKENIZERS_PARALLELISM"] = "true"
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.labelsize"] = 'large'


class WarmupDecayScheduler(_LRScheduler):
    """
    Scheduler implementing a linear warmup followed by a cosine annealing schedule.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        max_steps (int): Total number of training steps.
    """

    def __init__(self, optimizer, warmup_steps, max_steps):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing with restarts
            t = self.last_epoch - self.warmup_steps
            T = self.max_steps - self.warmup_steps
            return [base_lr * 0.5 * (1 + math.cos(math.pi * t / T)) for base_lr in self.base_lrs]


class BertSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super(BertSentimentClassifier, self).__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config.model["pretrained_model"])
        self.dense1 = nn.Linear(self.model.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256, 256)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(256, 1)

        tokenizer = PreTrainedTokenizer(Bunch(config.model))
        self.data_loader = TransformerDataLoader(config, tokenizer)
        self.loss = BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.confmat = ConfusionMatrix(num_classes=2, task="binary")
        self.batch_size = int(self.config.model["batch_size"])
        self.learning_rate = float(self.config.model['learning_rate'])
        self.num_epochs = int(self.config.model['epochs'])
        self.test_data = None
        self.train_data = None
        self.val_data = None
        self.val_labels = []
        self.val_predictions = []
        self.val_logits = []
        self.test_logits = []
        self.val_batch_sizes = []

    def prepare_data(self) -> None:
        self.train_data, self.val_data, self.test_data = self.data_loader.create_datasets()

    def train_dataloader(self) -> DataLoader:
        # NOTE: This function is needed for the Trainer class!
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.config.model["num_workers"],
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.config.model["num_workers"],
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.config.model["num_workers"],
            persistent_workers=True,
            pin_memory=True
        )

    def configure_optimizers(self) -> tuple[list[AdamW], list[LambdaLR]]:
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        # num_training_steps = len(self.train_data) * self.num_epochs
        # warmup_steps = int(num_training_steps * 0.1)  # number of warmup steps
        # scheduler = WarmupDecayScheduler(optimizer, warmup_steps, num_training_steps)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "frequency": 1
        #     }
        # }
        return optimizer

    def forward(
            self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output = self.model(token_ids, attention_mask)
        pooled_output = output[0][:, 0]
        dense1 = self.relu(self.dense1(pooled_output))
        drop1 = self.drop1(dense1)
        dense2 = self.relu(self.dense2(drop1))
        drop2 = self.drop2(dense2)
        logits = self.out(drop2)
        return logits

    def training_step(
            self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        token_ids, attention_mask, labels = batch
        labels = labels.float()
        logits = self.forward(token_ids, attention_mask)
        predictions = torch.sigmoid(logits).squeeze()

        loss = self.loss(logits.view(-1), labels.view(-1))

        train_acc = self.train_accuracy(predictions, labels.long())
        train_auroc = self.train_auroc(predictions, labels.long())
        self.log('loss', loss, on_step=True, on_epoch=False)
        self.log('accuracy', train_acc, on_step=True, on_epoch=False)
        self.log('weighted_auroc', train_auroc, on_step=True, on_epoch=False)
        # self.train_metrics_dict = {"loss": loss, "accuracy": self.train_accuracy, "weighted_auroc": self.train_auroc}

        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        token_ids, attention_mask, labels = batch
        logits = self.forward(token_ids, attention_mask)
        predictions = torch.sigmoid(logits).squeeze()
        return predictions

    def on_train_epoch_end(self) -> None:
        self.train_accuracy.reset()
        self.train_auroc.reset()

    def validation_step(
            self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        token_ids, attention_mask, labels = batch
        labels = labels.float()
        logits = self.forward(token_ids, attention_mask)

        predictions = torch.sigmoid(logits).squeeze()
        loss = self.loss(logits.view(-1), labels.view(-1))

        val_acc = self.val_accuracy(predictions, labels.long())
        val_auroc = self.val_auroc(predictions, labels.long())
        self.log('val_accuracy', val_acc, on_step=False, on_epoch=True)
        self.log('val_weighted_auroc', val_auroc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        # self.val_metrics_dict = {"loss": loss,
        #                          "accuracy": self.val_accuracy,
        #                          "weighted_auroc": self.val_auroc,
        #                          "predictions": predictions,
        #                          "labels": labels}

        self.val_predictions.append(predictions)
        self.val_logits.append(torch.sigmoid(logits))
        self.val_labels.append(labels)
        return predictions

    def on_validation_epoch_end(self) -> None:
        predictions = torch.cat(self.val_predictions)
        labels = torch.cat(self.val_labels)
        logits = torch.cat(self.val_logits)

        confmat = self.confmat(predictions.squeeze(), labels.long()).int().cpu().numpy()
        # scale predictions to [-1, 1] from [0, 1]
        predictions = predictions.unsqueeze(-1).cpu().numpy() * 2 - 1

        labels = labels.cpu().numpy()
        logits = logits.cpu().numpy()
        one_hot_labels = np.array([1 if item == 1 else -1 for item in labels])

        df_val = pd.DataFrame(np.concatenate([logits, predictions, one_hot_labels[..., None]], axis=1),
                              columns=['Probability', 'Predicted', 'Label'])

        pos_probas = df_val[df_val['Label'] == 1]
        neg_probas = df_val[df_val['Label'] == -1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        fig.suptitle('Model Probability Distribution on Validation Set')
        sp1 = sns.histplot(data=pos_probas, x="Probability", bins=20, ax=ax1)
        sp2 = sns.histplot(data=neg_probas, x="Probability", bins=20, ax=ax2)
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        ax1.set_title('Positive Predictions')
        ax2.set_ylabel('')
        ax2.set_xlabel('')
        ax2.set_title('Negative Predictions')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')
        plt.tight_layout()

        self.logger.log_table('Confusion Matrix', columns=['Predicted Positive', 'Predicted Negative'], data=confmat)
        self.logger.log_image('Prediction Probability Distribution', images=[fig])

        ax1.clear()
        ax2.clear()
        ax.clear()
        plt.close(fig)

        self.val_predictions.clear()
        self.val_labels.clear()
        self.val_logits.clear()
        self.val_auroc.reset()
        self.val_accuracy.reset()

    def test_step(
            self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        token_ids, attention_mask = batch
        logits = self.forward(token_ids, attention_mask)
        self.test_logits.append(torch.sigmoid(logits))
        return {"logits": logits}

    def on_test_epoch_end(self) -> None:
        logits = torch.cat(self.test_logits)

        ids = torch.arange(1, logits.shape[0] + 1).unsqueeze(-1)
        predictions = logits.round().cpu().int()
        logits = logits.cpu()
        ids = ids.cpu()
        predictions = 2 * predictions - 1

        logit_table = torch.cat((ids.float(), logits), dim=1).numpy()
        prediction_table = torch.cat((ids, predictions), dim=1).numpy()

        prediction_table_df = pd.DataFrame(prediction_table, columns=['Id', 'Prediction'])
        prediction_probabilities_df = pd.DataFrame(logit_table, columns=['Id', 'Probability'])

        prediction_table_df.to_csv('submission.csv', index=False)
        prediction_probabilities_df.to_csv('prediction_probabilities.csv', index=False)

        self.logger.log_table("prediction_probabilities", columns=['Id', 'Probability'], data=logit_table)
        self.logger.log_table("prediction_lables", columns=['Id', 'Prediction'], data=prediction_table)

        self.test_logits.clear()

    def select_data_from_entropy(self, logits):
        # compute shannon entropy
        entropy = -torch.sum(logits * torch.log(logits), dim=1)
        num_samples = torch.sum(torch.logical_and((logits > 0.3), (logits < 0.7))).item()
        # select data based on threshold of entropy
        _, indices = torch.topk(entropy, num_samples)

        return indices




