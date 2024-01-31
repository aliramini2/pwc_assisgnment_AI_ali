"""
This file will contain the definition of the FineTuner class,
responsible for initializing the model and tokenizer.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class FineTuner:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
