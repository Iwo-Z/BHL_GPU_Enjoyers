import ollama
from classifier import *
from textrank import *

class Evaluation(object):
    def __init__(self):
        self.model = DistilBertForSequenceClassification.from_pretrained("model")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer")

    def run_LLM(self, prompt):
        response = ollama.generate(
        model='mistral',
        prompt=prompt
        )
        return response['response']
    
    def run_textrank(self, prompt):
        tr = TextRankSummarizer()
        summary = tr.summarize(prompt)
        print("summary", summary)
        return self.run_LLM(summary)
    
    def run_optimized_LLM(self, prompt):
        cl = Classifier(self.model, self.tokenizer)
        class_ = cl.predict(prompt)
        print(f"Classified as: {class_}")

        if class_ == "prompt":
            return self.run_textrank(prompt)
        
        return "HARDCODED TEXT"