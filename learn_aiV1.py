import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import get_close_matches

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = nltk.WordNetLemmatizer()

# Load AI memory
try:
    with open("ai_memory.json", "r") as file:
        ai_memory = json.load(file)
except FileNotFoundError:
    ai_memory = {}

def preprocess_text(text):
    """Tokenize, remove stop words, and lemmatize"""
    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

def find_best_match(command):
    """Find the closest match in AI memory"""
    processed_command = preprocess_text(command)
    matches = get_close_matches(processed_command, ai_memory.keys(), n=1, cutoff=0.7)
    return matches[0] if matches else None

def train_ai(command, response):
    """Teach AI new responses"""
    processed_command = preprocess_text(command)
    ai_memory[processed_command] = response
    with open("ai_memory.json", "w") as file:
        json.dump(ai_memory, file)

def respond(command):
    """AI tries to respond based on learned commands"""
    match = find_best_match(command)
    if match:
        return ai_memory[match]
    else:
        return "I don't know that yet. Teach me!"

# Chat Loop
print("NLTK-Powered AI Assistant Ready! Type your commands.\n")

while True:
    user_input = input("You: ").strip().lower()
    
    if user_input in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = respond(user_input)

    if response == "I don't know that yet. Teach me!":
        new_response = input("Teach me the correct response: ").strip()
        train_ai(user_input, new_response)
        print("Got it! I'll remember that.")
    else:
        print("AI:", response)
