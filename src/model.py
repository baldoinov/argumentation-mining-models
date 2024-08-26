import torch
import lightning as pl

from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import AutoModelForSequenceClassification


class ArgumentationMiningTaskModel(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        learning_rate: float,
        model_checkpoint: str,
        id2label: dict,
        label2id: dict,
        class_weights: list,
        train_last_n_layers: int | str,
        task: str,
    ) -> None:
        super().__init__()

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            return_dict=True,
            num_labels=n_classes,
            id2label=id2label,
            label2id=label2id,
        )

        self.train_last_n_layers = train_last_n_layers
        parameters = list(self.bert.named_parameters())

        if self.train_last_n_layers == "full":
            pass
        else:
            for idx in range(len(parameters) - self.train_last_n_layers):
                param = parameters[idx][1]
                param.requires_grad = False
        
        self.learning_rate = learning_rate
        self.class_weights = torch.tensor(class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=n_classes)
        self.precision = Precision(task="multiclass", num_classes=n_classes)
        self.recall = Recall(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        logits = output.logits
        loss = 0

        if labels is not None:

            loss = self.criterion(input=logits, target=labels)

        return loss, logits

    def __common_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, logits = self.forward(input_ids, attention_mask, labels)

        return loss, logits, labels

    def training_step(self, batch, batch_idx):

        loss, logits, labels = self.__common_step(batch, batch_idx)
        predctions = torch.argmax(logits, dim=1)

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.accuracy(predctions, labels),
                "train_precision": self.precision(predctions, labels),
                "train_recall": self.recall(predctions, labels),
                "train_f1": self.f1_score(predctions, labels),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": loss, "logits": logits, "labels": labels}

    def validation_step(self, batch, batch_idx):

        loss, logits, labels = self.__common_step(batch, batch_idx)
        predctions = torch.argmax(logits, dim=1)

        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": self.accuracy(predctions, labels),
                "val_precision": self.precision(predctions, labels),
                "val_recall": self.recall(predctions, labels),
                "val_f1": self.f1_score(predctions, labels),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def test_step(self, batch, batch_idx):

        loss, logits, labels = self.__common_step(batch, batch_idx)
        predctions = torch.argmax(logits, dim=1)

        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": self.accuracy(predctions, labels),
                "test_precision": self.precision(predctions, labels),
                "test_recall": self.recall(predctions, labels),
                "test_f1": self.f1_score(predctions, labels),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, logits = self.forward(input_ids, attention_mask, labels)
        predctions = torch.argmax(logits, dim=1)

        return predctions

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
