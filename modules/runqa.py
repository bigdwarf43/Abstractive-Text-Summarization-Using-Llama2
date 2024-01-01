from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Qdrant

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


from langchain.text_splitter import CharacterTextSplitter
import os

import streamlit as st
from langchain.schema.document import Document


class runqa():

    def __init__(self, llm, text) -> None:
        self.llm = llm
        self.text = text

    def runChat(self):

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )

        # qaPromptText = """
        
        # """

        qaPromptText = PromptTemplate(
            input_variables=[
                "human_input",
                "chat_history",
                "context"
            ],
            template=(
                """
                [INST]
                <<SYS>>
                    You are a AI assistant whose expertise is reading and summarizing scientific papers.
                    You are given the chat history and a query.
                    Also attached, are the text embeddings that are most relevant to the document.
                    You are freely allowed to use / not use these to form your answer. The final answer is always dependant on you.
                    Skip the salutations and formalities, only output the answer.

                    {context}
                    History of the chat so far: 
                    {chat_history}

                    Query: 
                    {human_input}
                <</SYS>>

                Summary: [/INST]
                """
            )
        )

        chunks = text_splitter.split_text(self.text)

        # Declare the Memory
        # Should autoupdate once attached to the chain
        memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="human_input")

        # Load the qa chain
        # Type: stuff, stuffs the whole document as context in the prompt
        chain = load_qa_chain(self.llm, chain_type="stuff",
                              memory=memory, prompt=qaPromptText)

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

        if user_question and user_question != "":
            docs = knowledge_base.similarity_search(user_question, k=4)

            response = chain.run(
                input_documents=docs,
                human_input=user_question
            )


            if 'questions_history' in st.session_state:
                questions_history = st.session_state['questions_history']
                questions_history.append(user_question)
                st.session_state['questions_history'] = questions_history

                answers_history = st.session_state['answers_history']
                answers_history.append(response)
                st.session_state['answers_history'] = answers_history

                reference_history = st.session_state['reference_history']
                reference_history.append(docs)
                st.session_state['reference_history'] = reference_history

            else:
                questions_history = []
                questions_history.append(user_question)

                answers_history = []
                answers_history.append(response)

                reference_history = []
                reference_history.append(docs)

                st.session_state['questions_history'] = questions_history
                st.session_state['answers_history'] = answers_history
                st.session_state['reference_history'] = reference_history

            # st.subheader("Answer: ")
            # st.success(response)

            # st.subheader("Referenced Chunks: ")
            # for i in range(len(docs)):
            #      st.info(docs[i].page_content)

        if 'questions_history' in st.session_state:
            questions_history = st.session_state.questions_history
            answers_history = st.session_state.answers_history
            reference_history = st.session_state.reference_history
            for i in range(len(answers_history)):
   
                st.markdown(
                    f'<div class="user">{questions_history[i]}</div>', unsafe_allow_html=True)


                st.markdown(
                    f'<div class="ai">{answers_history[i]}</div>', unsafe_allow_html=True)
                with st.expander("Referenced chunks: "):
                    for j in reference_history[i]:
                        st.write(j.page_content)
