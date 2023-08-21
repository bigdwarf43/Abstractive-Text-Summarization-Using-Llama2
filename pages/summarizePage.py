import streamlit as st
from modules.summarizer import summarizer
from modules.Class import CustomLLM

from PyPDF2 import PdfReader



llm = CustomLLM(n=2)

#setup streamlit
st.set_page_config(page_title="Summarize your PDF")
st.header("summarize your PDF 💬")

#file uploader
pdfs = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)


if pdfs:
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    summarizeBtn = st.button("summarize")
    if summarizeBtn:
        summarizer = summarizer(llm=llm, text=text)
        output, summary_list = summarizer.summarize()

        st.success(output)
        st.write("Chunk wise summary")

        for doc in summary_list:
            st.info(doc)
