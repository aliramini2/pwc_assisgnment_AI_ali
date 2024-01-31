# This file contains the training and evaluation logic, including the computation of metrics.
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction

def train(model, train_dataset, eval_dataset, output_dir, num_train_epochs, batch_size, learning_rate, weight_decay):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy="epoch"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer

def evaluate(trainer, eval_dataset):
    return trainer.evaluate(eval_dataset)

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}
