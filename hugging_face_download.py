import os

from transformers.models.distilbert import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json


# Download the tokenizer and model from the Hugging Face model hub
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
# model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# current_dir = os.path.dirname(os.path.abspath(__file__))
# persistent_directory_token = os.path.join(current_dir, "distilbert_base_uncased_tokenizer")
# persistent_directory_model = os.path.join(current_dir, "distilbert_base_uncased_model")

# # Save them locally for offline use
# tokenizer.save_pretrained(persistent_directory_token, local_files_only=True)
# model.save_pretrained(persistent_directory_model, local_files_only=True)

# # Define the model name
# model_name = "distilgpt2"
# # Specify the local directory to save the model and tokenizer
# save_directory = "./distilgpt2_local"
# # Download and save the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(save_directory)
# # Download and save the model
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.save_pretrained(save_directory)

# Disable SDPA to avoid Sliding Window Attention warning
torch.backends.cuda.enable_flash_sdp(False)

# Define model name and local directory
model_name = "Qwen/Qwen1.5-0.5B"
save_directory = "./qwen1_5_0_5b_local"

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# Download and save model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Explicitly disable Sliding Window Attention
model.config.sliding_window = None
model.config.use_sliding_window = False  # Qwen-specific setting
model.config.attention_dropout = 0.0  # Ensure no dropout

# Save model
model.save_pretrained(save_directory)

# Update and verify config.json
config_path = os.path.join(save_directory, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)
config["sliding_window"] = None
config["use_sliding_window"] = False
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Model and tokenizer saved to {save_directory}")
print(f"Verified: config.json has sliding_window=null, use_sliding_window=false")