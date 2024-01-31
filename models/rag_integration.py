# This file contains the RAGChatbot class for integrating and using the RAG model.
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

class RAGChatbot:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=self.retriever)

    def get_response(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        response = self.model.generate(input_ids)
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
