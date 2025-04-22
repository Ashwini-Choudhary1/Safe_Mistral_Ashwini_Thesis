import torch
from model import model, tokenizer, CFG # Import initialized instances

def prepare_prompt(text):
    instructions = []
    if CFG.simplified_language:
        instructions.append("Keep the response simple and easy to understand.")
    if CFG.step_by_step_instructions:
        instructions.append("Provide step-by-step explanations.")
    if CFG.avoid_ambiguity:
        instructions.append("Avoid vague or ambiguous wording.")
    if CFG.provide_examples:
        instructions.append("Include relevant examples.")
    full_prompt = f"[INST] {text.strip()} {' '.join(instructions)} [/INST]"
    return full_prompt

def inference(prompt):
    if model is None or tokenizer is None:
        return "Model and/or tokenizer not loaded. Check logs for errors."
    try:
        encoded = tokenizer(prepare_prompt(prompt), return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = encoded.input_ids.to(device) # Use the 'device' variable
        attention_mask = encoded.attention_mask.to(device) # Use the 'device' variable

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output.split("[/INST]")[-1].strip() if "[/INST]" in output else output.strip()

        if CFG.use_bullet_points:
            response = response.replace("\n", "\n- ")
        if CFG.explicit_summaries:
            response += "\n\n**Summary:** Key points of the response."
        if CFG.consistent_tone:
            response = response.replace("!", ".")

        return response or "No meaningful response generated."
    except Exception as e:
        print(f"Error during inference: {e}")
        return "An error occurred during inference."
    finally:
        torch.cuda.empty_cache()
