#!/bin/bash
# ngrok_setup.sh

echo "Starting Gradio App..."
python /chatbot/interfaces/chatbot_interface.py &

echo "Launching Ngrok..."
ngrok http 7860  # Replace 7860 with the port used by your Gradio app
