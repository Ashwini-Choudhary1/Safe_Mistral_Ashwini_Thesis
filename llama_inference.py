import torch
from llama_model import model, tokenizer, CFG

def inference(prompt):
    if model is None or tokenizer is None:
        return "Llama model and/or tokenizer not loaded. Check logs for errors."
    try:
        encoded_input = tokenizer(
            CFG.llama_prompt_prefix + prompt + CFG.llama_prompt_suffix,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        input_ids = encoded_input.input_ids.to(model.device)
        attention_mask = encoded_input.attention_mask.to(model.device)

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text.split("[/INST]")[-1].strip() if "[/INST]" in output_text else output_text.strip()

        if CFG.use_bullet_points:
            response = response.replace("\n", "\n- ")
        if CFG.explicit_summaries:
            response += "\n\n**Summary:** Key points of the response."
        if CFG.consistent_tone:
            response = response.replace("!", ".")

        return response if response else "No meaningful response generated."
    except Exception as e:
        print(f"Error during Llama inference: {e}")
        return "An error occurred during Llama inference."
    finally:
        torch.cuda.empty_cache()
