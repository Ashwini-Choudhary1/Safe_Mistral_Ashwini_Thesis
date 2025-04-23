import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class CFG:
    ### Model
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    # You might need an access token if the model is private
    # hf_token = "YOUR_HUGGINGFACE_TOKEN"
    llama_prompt_prefix = "<s>[INST] "
    llama_prompt_suffix = " [/INST]"

    ### Autistic-Friendly Parameters (using a subset for brevity)
    use_bullet_points = True
    explicit_summaries = True
    consistent_tone = True

# Configure BitsAndBytes for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Llama model and tokenizer
try:
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        # token=CFG.hf_token  # Uncomment if needed
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        CFG.model_id,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
        # token=CFG.hf_token  # Uncomment if needed
    )
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Error loading Llama model: {e}")
    model = None
    tokenizer = None
