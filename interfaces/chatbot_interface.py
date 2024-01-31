import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

import gradio as gr
from models.chatbot_model import ChatbotModel
from utils.audio_transcription import AudioTranscriber
from models.rag_integration import RAGChatbot
from utils.prompt_optimization import PromptOptimizer

# Initialize the chatbot model and audio transcriber
chatbot = ChatbotModel(model_path="C:\\projects\\pwc\\results\\fine_tuned")
rag_chatbot = RAGChatbot()
prompt_optimizer = PromptOptimizer(chatbot)
audio_transcriber = AudioTranscriber()

def chatbot_response(text_input, audio_file):
    if text_input:
        optimized_prompt = prompt_optimizer.test_prompt(text_input)
        response = rag_chatbot.get_response(optimized_prompt)
    elif audio_file is not None:
        # Extract the audio data from the audio_file tuple
        _, audio_data = audio_file
        transcribed_text = audio_transcriber.transcribe(audio_data)
        optimized_prompt = prompt_optimizer.test_prompt(transcribed_text)
        response = rag_chatbot.get_response(optimized_prompt)
    else:
        response = "Please provide either text or audio input."
    return response

iface = gr.Interface(
    fn=chatbot_response,
    inputs=[
        gr.components.Textbox(lines=2, placeholder="Enter Text Here..."),
        gr.components.Audio()
    ],
    outputs="text"
)

iface.launch()
