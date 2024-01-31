from models.chatbot_model import ChatbotModel
from utils.audio_transcription import AudioTranscriber

# Load the model
model_path = './finetuned_model'
chatbot = ChatbotModel(model_path)

# Example text input test
text_input = "Hello, how are you?"
response = chatbot.get_response(text_input)
print("Response to text input:", response)

# Audio test
audio_transcriber = AudioTranscriber()
audio_input = "path_to_audio_file.wav"
transcribed_text = audio_transcriber.transcribe(audio_input)
response = chatbot.get_response(transcribed_text)
print("Response to audio input:", response)
