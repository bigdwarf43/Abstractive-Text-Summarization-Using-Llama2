import streamlit as st
from modules.summarizer import summarizer
from modules.Class import CustomLLM

from PyPDF2 import PdfReader

import evaluate

rouge = evaluate.load('rouge')



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
        output, summary_list, input_docs = summarizer.summarize()

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Source word count", value = str(len(text.split()))+" words")
        with c2:
            st.metric("Summary word count", value = str(len(output.split()))+" words")
        st.subheader("Summary: ")
        st.success(output)
        st.subheader("Chunk wise summary: ")

        for doc in summary_list:
            st.info(doc)

        st.subheader("Selected Chunks: ")
        for doc in input_docs:
            st.info(doc)

        # results = rouge.compute(predictions=[output],
        #                  references=[[x for x in summary_list]])


        # st.subheader("Rouge Scores: ")
        # st.success(f"Rouge2: {results['rouge2']}")
        # st.success(f"RougeL: {results['rougeL']}")
        # st.success(f"Rouge1: {results['rouge1']}")
        # st.success(f"RougeLsum: {results['rougeLsum']}")
