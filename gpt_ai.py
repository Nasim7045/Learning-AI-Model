import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import get_close_matches

# 🔹 Load GPT-2 Model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 🔹 Predefined Knowledge (GPT-2 handles general queries)
default_knowledge = {
    "who was gandhi": "Mahatma Gandhi was an Indian leader who fought for independence using non-violent resistance.",
    "who was einstein": "Albert Einstein was a physicist known for the theory of relativity and E=mc².",
    "what is coding": "Coding is writing instructions for computers using programming languages like Python, Java, or C++.",
    "why is eid celebrated": "Eid is a Muslim festival marking the end of Ramadan or Hajj.",
    "code in python for calculator": """def calculator(): 
    op = input("Enter operation (+, -, *, /): ")
    a = float(input("Enter first number: "))
    b = float(input("Enter second number: "))
    if op == '+': print(a + b)
    elif op == '-': print(a - b)
    elif op == '*': print(a * b)
    elif op == '/': print(a / b if b != 0 else "Cannot divide by zero.")
    else: print("Invalid operation")
calculator()"""
}

# 🔹 Load AI memory (learned knowledge)
try:
    with open("ai_memory.json", "r") as file:
        ai_memory = json.load(file)
except FileNotFoundError:
    ai_memory = {}

# Merge predefined knowledge with learned knowledge
ai_memory = {**default_knowledge, **ai_memory}

# 🔹 Store Question Learning Progress
question_frequency = {}

def preprocess_text(text):
    """Tokenize and clean user input"""
    words = word_tokenize(text.lower())
    return " ".join(words)

def find_best_match(command):
    """Find the closest stored answer"""
    processed_command = preprocess_text(command)
    matches = get_close_matches(processed_command, ai_memory.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def train_ai(command, response):
    """Teach AI new responses"""
    processed_command = preprocess_text(command)
    ai_memory[processed_command] = response
    question_frequency[processed_command] = question_frequency.get(processed_command, 0) + 1
    with open("ai_memory.json", "w") as file:
        json.dump(ai_memory, file)

def generate_gpt2_response(prompt):
    """Use GPT-2 to generate a response"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

def respond(command):
    """AI decides between stored knowledge and GPT-2"""
    match = find_best_match(command)

    if match:
        question_frequency[match] = question_frequency.get(match, 0) + 1
        return ai_memory[match] if question_frequency[match] < 3 else f"{ai_memory[match]} (I have learned this well now!)"
    
    # Use GPT-2 if no answer is found
    return generate_gpt2_response(command)

# 🔹 Chat Loop
print("🧠 GPT-2 AI Ready! Type your commands.\n")

while True:
    user_input = input("You: ").strip().lower()
    
    if user_input in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = respond(user_input)

    if response:
        print("AI:", response)
    else:
        new_response = input("I don't know that. Teach me: ").strip()
        train_ai(user_input, new_response)
        print("Got it! I'll remember that.")
