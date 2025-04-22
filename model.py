import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

class CFG:
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    hf_token = "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"  
    substrings_to_block = [
        "incite violence", "promote hatred", "terrorist act", "mass shooting",
        "bomb making", "self harm", "racial slurs", "hate speech", "sexist language",
        "homophobic remarks", "drug trafficking", "identity theft", "hacking tools",
        "child exploitation", "how to kill", "how to hurt", "making bombs",
        "creating weapons", "how to steal", "genocide", "torture", "bestiality",
        "pedophilia", "extremist ideology", "suicide methods", "disturbing gore",
        "promote anarchy", "recruit for terrorism", "white supremacy",
        "black supremacy", "any form of supremacy", "how to make explosives",
        "how to make poison", "how to make drugs",
    ]
    competitor_list = ["Apple", "Microsoft", "Azure", "AWS"]
    topics_list = [
        "violence", "hate speech", "graphic content", "illegal activities",
        "self-harm", "child exploitation", "extremism", "conspiracy theories",
        "misinformation", "medical misinformation", "financial fraud", "cults",
    ]

    verbose_output = True
    sensory_friendly_output = True
    simplified_language = True
    predictable_structure = True
    avoid_ambiguity = True
    provide_examples = True
    step_by_step_instructions = True
    avoid_metaphors = True
    use_bullet_points = True
    provide_definitions = True
    clear_transitions = True
    explicit_summaries = True
    consistent_tone = True
    visual_aids = False
    check_understanding = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=CFG.hf_token
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    CFG.model_id,
    trust_remote_code=True,
    padding_side="left",
    token=CFG.hf_token
)
tokenizer.pad_token = tokenizer.eos_token
