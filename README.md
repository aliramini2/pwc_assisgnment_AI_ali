# Chatbot Project

## Overview
This project is a chatbot application that utilizes a fine-tuned BERT model to understand and respond to user queries. It can process both text and audio inputs.

## Setup and Installation
To set up the chatbot project, follow these steps:

1. Clone the repository:
git clone [repository-url]

2. Install required dependencies:
pip install -r requirements.txt

## How to Run
To run the chatbot application, follow these steps:

1. To train the model:
python /chatbot/scripts/train_model.py

2. To launch the chatbot interface:
python /chatbot/interfaces/chatbot_interface.py


## Architecture
The project is structured as follows:

- `/models`: Contains the BERT model fine-tuning scripts.
- `/scripts`: Includes scripts for training and evaluating the model.
- `/data`: Stores the datasets used for training.
- `/utils`: Contains utility scripts for tokenization and other tasks.
- `/interfaces`: Holds the Gradio interface scripts.
- `/results`: Stores the fine-tuned models and TensorBoard logs.
- `/docker`: Docker configuration files.

## Contributors
- Ali AlRamini


