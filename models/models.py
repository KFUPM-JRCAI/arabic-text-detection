import sys
import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from transformers import (
    BertForSequenceClassification,
    XLMRobertaForSequenceClassification,
)


class BaseLitModel(LightningModule):
    def __init__(self, pad_token_id=0):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_id = int(pad_token_id)
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")
        self.train_accuracy = torchmetrics.Accuracy(task="binary")

    def step(self, batch, batch_idx, step_name):
        inputs, attention_mask, labels = batch
        outputs = self(inputs).squeeze()
        loss = F.binary_cross_entropy(outputs, labels.float())
        accuracy = getattr(self, f"{step_name}_accuracy")(
            outputs,
            labels,
        )
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_acc",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, "test")

    def configure_optimizers(self):
        raise NotImplementedError("Must be implemented in the subclass.")


class LitLSTMModel(BaseLitModel):
    def __init__(
        self,
        vocab_size,
        num_layers=3,
        rnn_hiddens=256,
        rnn_dropout=0.3,
        dropout_prob=0.45,
        embedding_size=512,
        learning_rate=0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=self.pad_token_id,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=rnn_hiddens,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features=rnn_hiddens, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.dropout_layer(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x[:, -1, :]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.75,
            patience=1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LitTransformerModel(BaseLitModel):
    def __init__(
        self,
        vocab_size,
        num_layers=3,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        max_length=512,
        learning_rate=0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=self.pad_token_id,
        )
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_length)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding_layer(x) * torch.sqrt(
            torch.tensor(
                self.hparams.d_model,
                dtype=torch.float32,
            ),
        )
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x[:, -1, :]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.75,
            patience=1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class LitBertModel(BaseLitModel):
    def __init__(
        self,
        learning_rate=1e-5,
        num_labels=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
        )
        self.fc = nn.Linear(self.num_labels, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize metrics
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")

        self.train_precision = torchmetrics.Precision(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")

        self.train_recall = torchmetrics.Recall(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")

        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        outputs = self.fc(outputs.logits)
        return self.sigmoid(outputs)

    def step(self, batch, batch_idx, step_name):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask).squeeze()
        loss = F.binary_cross_entropy(outputs, labels.float())

        # Calculate metrics
        accuracy = getattr(self, f"{step_name}_accuracy")(outputs, labels)
        precision = getattr(self, f"{step_name}_precision")(outputs, labels)
        recall = getattr(self, f"{step_name}_recall")(outputs, labels)
        f1 = getattr(self, f"{step_name}_f1")(outputs, labels)

        # Log metrics
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_f1",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.75,
            patience=1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class LitXLMRobertaModel(BaseLitModel):
    def __init__(
        self,
        learning_rate=1e-5,
        num_labels=2,
        model_name="xlm-roberta-base",  # Can be changed to "xlm-roberta-large" if needed
    ):
        super().__init__()
        if num_labels < 2:
            raise ValueError("Number of labels must be at least 2")
        self.num_labels = num_labels

        self.xlm_roberta = XLMRobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        if self.num_labels == 2:
            self.fc = nn.Linear(self.num_labels, 1)
            self.activation = nn.Sigmoid()
        else:
            self.fc = nn.Linear(self.num_labels, self.num_labels)
            self.activation = nn.Softmax(dim=1)

        # Initialize metrics based on the task type
        task = "binary" if num_labels == 2 else "multiclass"
        num_classes = num_labels if task != "binary" else None

        self.train_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes)

        self.train_precision = torchmetrics.Precision(
            task=task, num_classes=num_classes
        )
        self.val_precision = torchmetrics.Precision(task=task, num_classes=num_classes)
        self.test_precision = torchmetrics.Precision(task=task, num_classes=num_classes)

        self.train_recall = torchmetrics.Recall(task=task, num_classes=num_classes)
        self.val_recall = torchmetrics.Recall(task=task, num_classes=num_classes)
        self.test_recall = torchmetrics.Recall(task=task, num_classes=num_classes)

        self.train_f1 = torchmetrics.F1Score(task=task, num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task=task, num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task=task, num_classes=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)
        outputs = self.fc(outputs.logits)
        return self.activation(outputs)

    def step(self, batch, batch_idx, step_name):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        outputs = outputs.squeeze(-1)

        if self.num_labels == 2:
            loss = F.binary_cross_entropy(outputs, labels.float())
            predictions = (outputs > 0.5).float()
        else:
            loss = F.cross_entropy(outputs, labels.long())
            predictions = torch.argmax(outputs, dim=1)

        if step_name == "predict":
            return {"predictions": predictions, "labels": labels}

        # Calculate metrics
        accuracy = getattr(self, f"{step_name}_accuracy")(predictions, labels)
        precision = getattr(self, f"{step_name}_precision")(predictions, labels)
        recall = getattr(self, f"{step_name}_recall")(predictions, labels)
        f1 = getattr(self, f"{step_name}_f1")(predictions, labels)

        # Store the results
        step_results = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # Log metrics
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_acc",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{step_name}_f1",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return step_results

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "predict")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.75,
            patience=1,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
