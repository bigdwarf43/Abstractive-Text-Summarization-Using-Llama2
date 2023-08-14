
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings, GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant

from Class import CustomLLM

import streamlit as st
from PyPDF2 import PdfReader

llm = CustomLLM(n=10)



callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

chain = load_qa_chain(llm, chain_type="stuff")

if "Helpful Answer:" in chain.llm_chain.prompt.template:
        chain.llm_chain.prompt.template = (
            f"### Human:{chain.llm_chain.prompt.template}".replace(
                "Helpful Answer: You are a AI assistant whose expertise is reading and summarizing scientific papers. You are given a query, a series of text embeddings and the title from a paper in order of their cosine similarity to the query. You must take the given embeddings and return a very detailed summary of the paper in the languange of the query:", "\n### Assistant:"
            )
        )

st.set_page_config(page_title="Ask your PDF")
st.header("Ask your PDF 💬")
pdfs = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)

if pdfs:
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
            )
        chunks = text_splitter.split_text(text)

    embeddings = GPT4AllEmbeddings()
    #embeddings = SentenceTransformerEmbeddings(model_name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")

    knowledge_base = Qdrant.from_texts(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )

    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        docs = knowledge_base.similarity_search(user_question, k=4)

        # Calculating prompt (takes time and can optionally be removed)
        # prompt_len = chain.prompt_length(docs=docs, question=user_question)
        # st.write(f"Prompt len: {prompt_len}")
        # if prompt_len > llm.n_ctx:
        #     st.write(
        #         "Prompt length is more than n_ctx. This will likely fail. Increase model's context, reduce chunk's \
        #             sizes or question length, or retrieve less number of docs."
        #     )
        print(st.write(docs))
        # Grab and print response
        response = chain.run(input_documents=docs, question=user_question)
        st.write(response)
