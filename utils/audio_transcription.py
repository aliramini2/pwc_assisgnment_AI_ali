import speech_recognition as sr
import io

class AudioTranscriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe(self, audio_data):
        audio_file = io.BytesIO(audio_data)
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service"
        except ValueError:
            return "Unsupported audio file format"

# Example usage
# transcriber = AudioTranscriber()
# print(transcriber.transcribe(audio_data))
