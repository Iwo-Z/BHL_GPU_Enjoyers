from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

class Classifier:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, prompt):
        prompt = [prompt]

        encodings = self.tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

        class_names = ["greetings", "thanking", "goodbye", "prompt"]
        return class_names[predicted_class_id]

# EXAMPLE USAGE:
#
# model = DistilBertForSequenceClassification.from_pretrained("model")
# tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer")
#
# from classifier import *
# c = Classifier(model, tokenizer)
# c.predict("thanks")