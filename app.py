from modules.Class import CustomLLM

import streamlit as st
from PyPDF2 import PdfReader

from modules.runqa import runqa

#initialize the llm
llm = CustomLLM(n=2)

#setup streamlit
st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 💬")

# Linking the styles
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#file uploader
pdfs = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)

if pdfs:
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    runqa = runqa(llm=llm, text=text)
    runqa.runChat()


# st.markdown(f'<div class="user">User question goes here</div>', unsafe_allow_html=True)
# st.markdown(f'<div class="ai">AI response goes here</div>', unsafe_allow_html=True)
