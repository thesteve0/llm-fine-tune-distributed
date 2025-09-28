#!/usr/bin/env python3
"""
Simple CLI tool to ask questions to the fine-tuned SmolLM3-3B Q&A model.

Usage:
    python ask_tuned_model.py "How do I check my car's oil level?"
    python ask_tuned_model.py "How do I fix a leaky faucet?"
"""
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path="outputs/best_model"):
    """Load the fine-tuned model and tokenizer."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Make sure you have trained and saved the model first.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def ask_question(model, tokenizer, question):
    """Ask a question to the model and return the response."""
    messages = [
        {"role": "system", "content": "You are a wilderness survival and practical skills expert. Your mission is to provide comprehensive, detailed guidance on essential survival and practical skills. Give thorough, step-by-step instructions with explanations of why each step matters.\n\nYour expertise covers:\n- Wilderness Survival Basics: Rule of 3s (3 minutes without air, 3 hours without shelter in harsh conditions, 3 days without water, 3 weeks without food), emergency signaling techniques, essential knots, identifying poisonous plants and safe alternatives\n- Basic First Aid: Treatment for cuts, burns, sprains, shock, and emergency care procedures\n- Simple Car Maintenance: Checking fluids (oil, coolant, brake, transmission), tire inspection and pressure, lights and electrical systems\n- Basic Cooking Techniques: Food safety, preparation methods, cooking over open fires, food preservation\n- Common Measurement Conversions: Imperial to metric, cooking measurements, distance and weight conversions\n- Essential Knots: Bowline, clove hitch, trucker's hitch, figure-eight, sheet bend, and their practical applications\n\nAlways provide detailed explanations, safety warnings when relevant, and multiple approaches when possible. Your responses should be comprehensive enough to help someone learn and apply these skills safely and effectively. Aim for thorough, educational responses rather than brief answers."},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3768,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Find the assistant marker and extract everything after it
    possible_markers = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "assistant\n",
        "assistant:",
        "<|assistant|>"
    ]

    response = None
    for marker in possible_markers:
        if marker in full_response:
            start_idx = full_response.find(marker) + len(marker)
            assistant_response = full_response[start_idx:]

            if "<|im_end|>" in assistant_response:
                assistant_response = assistant_response.split("<|im_end|>")[0]

            response = assistant_response.strip()
            break

    if response is None:
        response = full_response[len(prompt):].strip()

    return response


def main():
    """Main function to handle command line arguments and run the Q&A."""
    if len(sys.argv) < 2:
        print("Usage: python ask_tuned_model.py \"Your question here\"")
        print("\nExamples:")
        print('  python ask_tuned_model.py "How often should I check my oil level?"')
        print('  python ask_tuned_model.py "How do I fix a leaky faucet?"')
        print('  python ask_tuned_model.py "What should I do if my computer won\'t start?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    model, tokenizer = load_model()

    try:
        answer = ask_question(model, tokenizer, question)
        print(f"Answer:")
        print(answer)
    except Exception as e:
        print(f"Error generating response: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()