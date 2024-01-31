# The main script to run your model training and evaluation.

import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from models.bert_model import FineTuner
from utils.data_processor import load_and_tokenize_dataset
from scripts.train_eval import train, evaluate

model_name = "bert-base-uncased"
fine_tuner = FineTuner(model_name=model_name, num_labels=4)

tokenized_dataset = load_and_tokenize_dataset(fine_tuner.tokenizer, "ag_news")

train_dataset = tokenized_dataset["train"].shuffle().select(range(2000)) # small traning samples
eval_dataset = tokenized_dataset["test"].shuffle().select(range(1000)) # small testing samples

trainer = train(
    model=fine_tuner.model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    output_dir="./pwc/results/fine_tuned",
    num_train_epochs=5, # 1 epoch to speedup the training process
    batch_size=8,
    learning_rate=2e-4,
    weight_decay=0.01,
)
trainer.save_model("./pwc/results/fine_tuned")
evaluation_results = evaluate(trainer, eval_dataset)
print(evaluation_results)
