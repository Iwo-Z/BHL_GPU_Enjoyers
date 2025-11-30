from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.classifier import *
from utils.textrank import *
import torch

class Evaluation(object):
    def __init__(self):
        self.model = DistilBertForSequenceClassification.from_pretrained("model")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer")

    def run_LLM(self, prompt):
        tokenizerLLM = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        modelLLM = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            device_map="auto",
            load_in_4bit=True,
            dtype=torch.float16
        )

        inputs = tokenizerLLM(prompt, return_tensors="pt").to(modelLLM.device)
        output = modelLLM.generate(**inputs, max_new_tokens=100)
        return tokenizerLLM.decode(output[0], skip_special_tokens=True)
    
    def run_textrank(self, prompt):
        tr = TextRankSummarizer()
        summary = tr.summarize(prompt)
        print("summary", summary)
        return self.run_LLM(summary)
    
    def run_optimized_LLM(self, prompt):
        cl = Classifier(self.model, self.tokenizer)
        class_ = cl.predict(prompt)
        print(f"Classified as: {class_}")

        if class_ == "greetings":
            return "Hello there! How can I help you today?"
        elif class_ == "thanking":
            return "No problem! If you have any more questions, feel free to ask."
        elif class_ == "goodbye":
            return "Goodbye! Have a great day!"
        elif class_ == "prompt": 
            return self.run_textrank(prompt)
        
        return "error"