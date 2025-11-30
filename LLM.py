import ollama


response = ollama.generate(
    model='mistral',
    prompt='Explain machine learning in simple terms'
)
print(response['response'])
