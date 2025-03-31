import os
import subprocess
import torch
import nltk
import pandas as pd
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, 
                          DataCollatorForLanguageModeling)
from datasets import load_dataset, Dataset

# Ensure necessary libraries are installed
def install_packages():
    packages = ["torch", "transformers", "datasets", "nltk", "pandas"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run(["pip", "install", package], check=True)

install_packages()

# Ensure NLTK tokenizer data is available
nltk.download('punkt')

# Load dataset from Hugging Face (10% of data to reduce memory usage)
try:
    dataset = load_dataset("bigcode/the-stack-smol", split="train[:10%]")  # Load only 10%
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Preprocessing function
def preprocess_data(example):
    return {"text": example["content"]}

# Apply preprocessing
dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names)

# Tokenize using NLTK
sentences = [nltk.sent_tokenize(text) for text in dataset["text"]]
flattened_sentences = [sent for sublist in sentences for sent in sublist]

# Create new dataset
dataset = Dataset.from_dict({"text": flattened_sentences})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="longest", truncation=True)

# Apply tokenization with batching
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator for training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load GPT-2 model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Train the model
print("Training the model...")
trainer.train()

# Text generation function
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(
        **inputs,
        max_length=100,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.7,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Interactive Q&A Loop
while True:
    user_input = input("Ask me a programming question (or type 'exit' to quit): ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    response = generate_text(user_input)
    print(f"AI: {response}\n")
