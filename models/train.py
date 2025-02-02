# This is a sample file on how to train a model

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import sys

sys.path.append(os.getcwd())  # add the root folder to path
import logging

from data import AraSumDataModule
from models import LitLSTMModel, LitTransformerModel, LitBertModel, LitXLMRobertaModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    RichProgressBar,
    ModelCheckpoint,
    EarlyStopping,
)

# Initialize the data module
# data_module = HC3TextDataModule()
data_module = AraSumDataModule(generated_by="llama-batched")

data_module.setup()

# Define the model (you can switch between different models)

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

models = {
    "lstm": LitLSTMModel(vocab_size=data_module.tokenizer.vocab_size),
    "transformer": LitTransformerModel(vocab_size=data_module.tokenizer.vocab_size),
    "bert": LitBertModel(),
    "roberta": LitXLMRobertaModel(),
}

model = models["roberta"]


# # Add EarlyStopping and ModelCheckpoint callbacks
early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    min_delta=0.0,  # Minimum change to qualify as improvement
    patience=3,  # Stop training after this epochs without improvement
    verbose=True,  # Print information at each validation step
    mode="min",  # Mode to minimize the monitored metric
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Monitor validation loss
    dirpath="trained_detectors/Arabic/LlamaArabicAIDetector/checkpoints",  # Directory to save checkpoints
    filename="best-checkpoint",  # Filename for the best checkpoint
    save_top_k=1,  # Save only the best checkpoint
    mode="min",  # Mode to minimize the monitored metric
)

# # Initialize a trainer with callbacks
trainer = pl.Trainer(
    devices=1,
    max_epochs=100,
    accelerator="auto",
    val_check_interval=0.25,
    check_val_every_n_epoch=1,
    callbacks=[RichProgressBar(), early_stopping_callback, checkpoint_callback],
)

# # Train the model
# trainer.fit(model, datamodule=data_module)
# # Test the model using the best checkpoint

model = model.__class__.load_from_checkpoint(
    "trained_detectors/Arabic/LlamaArabicAIDetector/checkpoints/best-checkpoint.ckpt"
)

trainer.test(
    model,
    datamodule=data_module,
    # ckpt_path=checkpoint_callback.best_model_path,
    # ckpt_path="trained_detectors/Arabic/LLamaArabicAIDetector/checkpoints/best-checkpoint.ckpt",
)
