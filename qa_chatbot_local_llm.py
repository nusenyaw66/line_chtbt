import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize HuggingFace embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Load the Chroma vector store
vector_store_path = "./db/hugging_face_chroma_with_metadata"  # Path to the Chroma vector store
vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)

# Initialize HuggingFace model and tokenizer
persistent_directory = "/Users/wsun/Programming/local_llm/qwen1_5_0_5b_local"
tokenizer = AutoTokenizer.from_pretrained(persistent_directory)
model = AutoModelForCausalLM.from_pretrained(persistent_directory)

# Disable Sliding Window Attention to avoid SDPA warning
model.config.sliding_window = None  # Disable SWA
model.config.attention_dropout = 0.0  # Optional: Ensure no dropout for consistency

# Set pad_token to eos_token and ensure proper padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Use left padding for generation

# Move model to MPS device if available（for MacOS with M1/M2 chips）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Function to retrieve relevant documents
def retrieve_documents(query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return results

# Function to generate QA prompt
def generate_qa_prompt(query, retrieved_docs):
    context = "\n".join([f"Q: {doc.metadata['question']}\nA: {doc.page_content}" for doc in retrieved_docs])
    prompt = f"""你是一個繁體中文問答聊天機器人。請根據以下上下文或你的知識回答使用者的問題。如果上下文無相關資訊，根據你的知識提供答案；若仍不知道，說不知道。

    
Context:
{context}

User Question: {query}

Answer: """
    return prompt

# Function to generate answer
def generate_answer(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,  # Increased to handle longer prompts
        padding=True,
        return_attention_mask=True
    ).to(device)
    # Debug: Print input details
    input_length = inputs["input_ids"].shape[1]
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    print(f"Input length: {input_length} tokens")
    print(f"Input text: {input_text}")
    
    # Check if input + max_new_tokens exceeds model limit (1024 tokens)
    max_new_tokens = 300  # Increased to allow more output
    if input_length + max_new_tokens > 1024:
        max_new_tokens = 1024 - input_length
        print(f"Adjusted max_new_tokens to {max_new_tokens} to stay within 1024-token limit")
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    # Debug: Print raw output details
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Raw output length: {len(outputs[0])} tokens")
    print(f"Raw output text: {output_text}")
    
    # Extract answer, handling short outputs
    answer = output_text[len(prompt):].strip() if len(output_text) > len(prompt) else output_text.strip()
    # Debug: Print extracted answer
    print(f"Extracted answer: {answer}")
    return answer

def qa_line_chatbot(query):
    try:
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(query)
            
        prompt = generate_qa_prompt(query, retrieved_docs)
        answer = generate_answer(prompt)

        # Save to chat history
        # chat_history.append((query, answer))

        return (answer)
    except Exception as e:
        print(f"Error in qa_line_chatbot: {e}")
        return "抱歉，無法處理您的請求，請稍後再試。"
    
# Function for continuous chatbot interaction
def qa_chatbot():
    print("Welcome to the QA Chatbot! Enter your question or type 'exit' to quit.")
    while True:
        query = input("Your question: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid question.")
            continue
        
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(query)
        # Debug: Print retrieved documents
        print("Retrieved documents:")
        for doc in retrieved_docs:
            print(f"Metadata: {doc.metadata}, Content: {doc.page_content}")
        
        # Generate QA prompt
        prompt = generate_qa_prompt(query, retrieved_docs)
        # Debug: Print prompt details
        prompt_tokens = len(tokenizer.encode(prompt))
        print(f"Prompt length: {prompt_tokens} tokens")
        print(f"Prompt text: {prompt}")
        
        # Generate answer
        answer = generate_answer(prompt)
        print(f"Query: {query}")
        print(f"Answer: {answer}\n")

# Run the chatbot
if __name__ == "__main__":
    qa_chatbot()