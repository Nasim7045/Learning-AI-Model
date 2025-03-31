from transformers import pipeline

# Load GPT-2 Model
generator = pipeline("text-generation", model="gpt2")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    response = generator(user_input, max_length=100, num_return_sequences=1)
    print("AI:", response[0]["generated_text"])
