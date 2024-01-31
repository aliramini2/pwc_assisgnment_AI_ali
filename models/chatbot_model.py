# chatbot_interface.py
import sys
import os
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Adjust these paths as necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_transcription import AudioTranscriber
from utils.text_utils import preprocess_input, process_response  # Ensure these are implemented

class ChatbotModel:
    def __init__(self, model_path, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.labels = ["World", "Sports", "Business", "Sci/Tech"]  # Adjust labels as per your dataset

    def get_response(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        label_index = outputs.logits.argmax(-1).item()  # Get the index of the max logit
        return self.labels[label_index]  # Map index to label name

# Initialize the chatbot model and audio transcriber
chatbot_model = ChatbotModel(model_path="./results/fine_tuned")
audio_transcriber = AudioTranscriber()

def chatbot_response(text_input, audio_file):
    if text_input:
        processed_text = preprocess_input(text_input)  # Preprocess text input
        raw_response = chatbot_model.get_response(processed_text)
        response = process_response(raw_response)  # Process model response
    elif audio_file is not None:
        # Extract the audio data from the audio_file tuple
        _, audio_data = audio_file
        # Transcribe the audio data
        transcribed_text = audio_transcriber.transcribe(audio_data)
        processed_text = preprocess_input(transcribed_text)  # Preprocess transcribed text
        raw_response = chatbot_model.get_response(processed_text)
        response = process_response(raw_response)  # Process model response
    else:
        response = "Please provide either text or audio input."
    return response

# Define Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs=[gr.Textbox(lines=2, placeholder="Enter Text Here..."), gr.Audio()],
    outputs=gr.Text()
)

if __name__ == "__main__":
    iface.launch()
