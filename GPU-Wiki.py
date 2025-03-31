import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 📌 Step 1: Install missing dependencies
try:
    import transformers
    import datasets
except ImportError:
    os.system("pip install transformers datasets torch")

# 📌 Step 2: Load Wikipedia Dataset (Smaller Subset for Faster Training)
print("Downloading Wikipedia dataset...")
dataset = load_dataset("wikipedia", "20220301.en", split="train[:5%]", trust_remote_code=True)

# 📌 Step 3: Standardize Dataset Format
print("Wikipedia dataset columns:", dataset.column_names)
dataset = dataset.map(lambda x: {"text": str(x["text"])}).remove_columns(["id", "url", "title"])

# 📌 Step 4: Load DistilGPT-2 Tokenizer & Model
print("Loading DistilGPT-2 model...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure tokenizer padding is set correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 📌 Step 5: Tokenize Data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 📌 Step 6: Setup Training (Optimized for CPU)
training_args = TrainingArguments(
    output_dir="./distilgpt2-finetuned",
    per_device_train_batch_size=4,  # Increased batch size for better CPU utilization
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    num_train_epochs=3,  # Fine-tune for 3 epochs
    load_best_model_at_end=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 📌 Step 7: Train the Model
print("Training DistilGPT-2 on Wikipedia data...")
trainer.train()

# 📌 Step 8: Save Model
model.save_pretrained("./distilgpt2-finetuned")
tokenizer.save_pretrained("./distilgpt2-finetuned")

# 📌 Step 9: Chatbot Loop
print("\n✅ Training Complete! AI is Ready. Type 'exit' to stop.")
generator = pipeline("text-generation", model="./distilgpt2-finetuned", tokenizer=tokenizer)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    response = generator(user_input, max_length=100, num_return_sequences=1)
    print("AI:", response[0]["generated_text"])
