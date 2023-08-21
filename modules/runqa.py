from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Qdrant

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain.text_splitter import CharacterTextSplitter


import streamlit as st

class runqa():
    
    def __init__(self, llm, text) -> None:
         self.llm = llm
         self.text = text

    def runChat(self):

        text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
            )
        
        qaPromptText = """
        [INST]
        <<SYS>>
            You are a AI assistant whose expertise is reading and summarizing scientific papers. 
            You are given a query, a series of text embeddings from a paper in order of their cosine similarity to the query. 
            You must take the given embeddings and return a very detailed summary of the paper in the languange of the query.
        <</SYS>>

        Summary: [/INST]
        """
        chunks = text_splitter.split_text(self.text)
        #Load the qa chain
        #Type: stuff, stuffs the whole document as context in the prompt 
        chain = load_qa_chain(self.llm, chain_type="stuff")
        if "Helpful Answer:" in chain.llm_chain.prompt.template:
                chain.llm_chain.prompt.template = (
                    f"### Human:{chain.llm_chain.prompt.template}".replace(
                        "Helpful Answer: You are a AI assistant whose expertise is reading and summarizing scientific papers. You are given a query, a series of text embeddings from a paper in order of their cosine similarity to the query. You must take the given embeddings and return a very detailed summary of the paper in the languange of the query.:", "\n### Assistant:"
                    )
                )

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

            st.write(docs)

            # Grab and print response
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)

     