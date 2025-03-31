import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK tokenizer data is available
nltk.download('punkt')

# Improved dataset with clearer question-answer formatting
data = """
What is the time complexity of QuickSort? The average case time complexity of QuickSort is O(n log n).
Who was the first president of the United States? The first president of the United States was George Washington.
What is the purpose of a for loop in programming? A for loop is used to iterate over a sequence of elements.
When did World War II start? World War II started in 1939.
When did World War I start? World War I started in 1914.
What is a variable in programming? A variable is a named storage location in memory that holds a value.
Who developed Python? Python was developed by Guido van Rossum in 1991.
What is 5 + 2? 5 + 2 equals 7.
"""

# Tokenizing dataset using NLTK
sentences = sent_tokenize(data)
dataset = Dataset.from_dict({"text": sentences})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Load model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))  # Ensure token embeddings are updated

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Function to generate text from the trained model
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(
        **inputs,
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,  # Enables randomness for more diverse responses
        top_k=40,  # Limits vocabulary selection for better coherence
        top_p=0.9,  # Nucleus sampling for diversity
        temperature=0.7,  # Adjusts randomness (lower = more deterministic)
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Interactive question-answering
while True:
    user_input = input("Ask me a question (or type 'exit' to quit): ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    response = generate_text(user_input)
    print(f"AI: {response}\n")
