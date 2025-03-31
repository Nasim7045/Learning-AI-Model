import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Step 1: Load a Small Wikipedia Subset
print("📥 Downloading Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", split="train[:200]")  # ✅ Load only first 200 samples

# Step 2: Clean Dataset
print("📑 Standardizing dataset...")
dataset = dataset.map(lambda x: {"text": str(x["text"])}).remove_columns(["id", "url", "title"])

# Step 3: Load Tokenizer & Model
print("🛠️ Loading DistilGPT-2 model...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 4: Tokenize Data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])  # ✅ Removes unused columns

# Step 5: Training Arguments (Optimized for CPU)
training_args = TrainingArguments(
    output_dir="./distilgpt2-finetuned",
    per_device_train_batch_size=2,  # ✅ Reduce batch size for CPU
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    remove_unused_columns=False,  # ✅ Prevent auto-removal of required columns
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Step 6: Train Model
print("🚀 Training DistilGPT-2...")
trainer.train()

# Step 7: Save Model
model.save_pretrained("./distilgpt2-finetuned")
tokenizer.save_pretrained("./distilgpt2-finetuned")

# Step 8: Chatbot Loop
print("\n✅ Training Complete! AI is Ready. Type 'exit' to stop.")
from transformers import pipeline
generator = pipeline("text-generation", model="./distilgpt2-finetuned", tokenizer=tokenizer)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = generator(user_input, max_length=100, num_return_sequences=1)
    print("AI:", response[0]["generated_text"])
