import os
import torch
import nltk
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, 
                          DataCollatorForLanguageModeling)
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# Ensure NLTK tokenizer data is available
nltk.download('punkt')

# Load dataset (Reduce size to speed up training)
try:
    dataset = load_dataset("bigcode/the-stack-smol", split="train[:5000]")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Preprocessing function
def preprocess_data(example):
    return {"text": example["content"]}

dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names)

# Tokenize using NLTK (Limit to 3000 sentences for speed)
sentences = [nltk.sent_tokenize(text) for text in dataset["text"]]
flattened_sentences = [sent for sublist in sentences for sent in sublist]
sampled_sentences, _ = train_test_split(flattened_sentences, train_size=3000, random_state=42)

dataset = Dataset.from_dict({"text": sampled_sentences})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load GPT-2 model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Training arguments (Optimized for speed)
training_args = TrainingArguments(
    output_dir="./results",
    max_steps=100,  # Reduce steps to speed up training
    per_device_train_batch_size=16,  # Keep batch size reasonable
    fp16=True,  # Faster training if GPU is available
    save_strategy="no",
    logging_strategy="no",
)


# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Train the model
print("Training the model... (Should take ~2 minutes)")
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
