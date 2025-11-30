# import ollama

# response = ollama.generate(
#     model='mistral',
#     prompt='Explain machine learning in simple terms'
# )
# print(response['response'])

from classifier import *

model = DistilBertForSequenceClassification.from_pretrained("model")
tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer")

cl = Classifier(model, tokenizer)
result = cl.predict("bye friend")
print("Prediction:", result)