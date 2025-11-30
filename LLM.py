from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",          # automatically puts layers on GPU/CPU
    load_in_4bit=True,          # enable 4-bit quantization
    dtype=torch.float16          # computation in float16 for better numeric stability
)


prompt = """Write a program in assembly language to calculate the sum of two numbers, where the numbers are stored in memory locations. The program should meet the following requirements:

Declare and initialize two variables x and y with the values 27 and 11, respectively.
Reserve exactly 10 memory locations to store the variables x and y.
Load the value of x into a register.
Load the value of y into a register.
Add the contents of the two registers.
Store the result in a memory location.
Print the result in hexadecimal format.
Implement error handling for overflow situations, where the sum exceeds the maximum value that can be stored in a register.
Use conditional branching instructions to handle the error situation appropriately.
Display an error message if an overflow occurs, indicating that the sum cannot be calculated accurately.
Test your program with the given values for x and y (27 and 11) and verify that the correct sum is calculated and printed"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


output = model.generate(**inputs, max_new_tokens=100)


print(tokenizer.decode(output[0], skip_special_tokens=True))