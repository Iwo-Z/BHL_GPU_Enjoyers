import ollama


response = ollama.generate(
    model='gemma3:4b',
    prompt='Explain machine learning in simple terms'
)
print(response['response'])
