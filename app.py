import streamlit as st
from inference import inference
from safeguards import scan_input, scan_output

st.title("🕵️ Prompt Safety Checker")
user_input = st.text_area("Enter your prompt:")

if st.button("Submit"):
    input_text, is_valid, _ = scan_input(None, user_input)
    if not is_valid:
        st.error(input_text)
    else:
        with st.spinner("Running inference..."):
            output = inference(input_text)
            response, valid, _ = scan_output(None, input_text, output)
            if not valid:
                st.warning(response)
            else:
                st.markdown(response)
