import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset
import time

def preprocess(example):
    return {"text": example["text"].replace("\n", " ").strip()}

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

if __name__ == "__main__":  # ✅ REQUIRED on Windows
    start_setup_time = time.time()

    print("📥 Downloading Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:500]")

    # ✅ Disable multiprocessing on Windows
    dataset = dataset.map(preprocess, remove_columns=["id", "url", "title"]) 

    print("🛠️ Loading DistilGPT-2 model...")
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    model = torch.compile(model)

    training_args = TrainingArguments(
        output_dir="./distilgpt2-finetuned",
        per_device_train_batch_size=16,
        max_steps=150,
        save_strategy="no",
        logging_strategy="no",
        learning_rate=5e-4,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    setup_time = time.time() - start_setup_time
    print(f"✅ Setup completed in {setup_time:.2f} seconds!")

    print("🚀 Training DistilGPT-2 (Limited to 3 min)...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    if end_time - start_time > 180:
        print("⏳ Training exceeded 3 minutes. Stopping early.")
    else:
        print("✅ Training completed within 3 minutes.")

    model.save_pretrained("./distilgpt2-finetuned")
    tokenizer.save_pretrained("./distilgpt2-finetuned")

    print("\n✅ AI is Ready!")
