# Contains logic for testing and evaluating different prompt structures.
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)


from models.chatbot_model import ChatbotModel

class PromptOptimizer:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def test_prompt(self, prompt):
        # This method would test the given prompt with the chatbot model and return the response.
        # The response can then be evaluated for relevance, coherence, and informativeness.
        response = self.chatbot.get_response(prompt)
        return response

    def evaluate_prompt(self, response):
        # Implement logic to evaluate the response.
        # This could involve manual evaluation or specific metrics.
        pass

# Example usage
if __name__ == "__main__":
    model_path = "C:\\projects\\pwc\\results\\fine_tuned"  # Adjust path as needed
    chatbot = ChatbotModel(model_path)
    prompt_optimizer = PromptOptimizer(chatbot)

    test_prompt = "What's the latest news in technology?"
    response = prompt_optimizer.test_prompt(test_prompt)
    print("Response:", response)