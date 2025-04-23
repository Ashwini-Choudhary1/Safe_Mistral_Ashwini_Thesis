import streamlit as st
from inference import inference as mistral_inference
from safeguards import scan_input, scan_output
from llama_inference import inference as llama_inference

st.title("🤖 Model Selection & Prompt Safety Checker")

model_choice = st.radio("Choose a Language Model:", ("Mistral", "Llama"))

if model_choice == "Mistral":
    st.info("Mistral Prompt Structure: Simply enter your prompt.")
elif model_choice == "Llama":
    st.info("Llama Prompt Structure: Enter your prompt, and optional instructions like 'Use simple language.' will be added.")

user_input = st.text_area("Enter your prompt:")

if st.button("Submit"):
    input_text, is_valid, _ = scan_input(None, user_input)
    if not is_valid:
        st.error(input_text)
    else:
        with st.spinner(f"Running inference with {model_choice}..."):
            if model_choice == "Mistral":
                output = mistral_inference(input_text)
            elif model_choice == "Llama":
                output = llama_inference(user_input)  # Llama's inference includes prompt prep

            response, valid, _ = scan_output(None, input_text, output)
            if not valid:
                st.warning(response)
            else:
                st.markdown(response)
