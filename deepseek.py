# run_chat.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from install_dependencies import load_model

def chat_with_model(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", max_length=500):
    print("Loading model...")
    tokenizer, model = load_model(model_name)
    
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(**inputs, max_length=max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Bot:", response, "\n")

if __name__ == "__main__":
    chat_with_model()
