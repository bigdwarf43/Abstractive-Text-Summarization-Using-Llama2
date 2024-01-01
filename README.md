# Abstractive Summarization of Long Texts And a Retrieval Augmented Generation Pipeline using Llama2 Large Language Model
Our project allows users to input multiple PDFs to generate a concise summary that encompasses key points from all documents. ##
Users can also engage in question-answering with the PDFs using the implemented Retrieval Augmented Generation workflow.

Our pipeline uses Langchain, Qdrant vector database and the LLama2 large language model.
The Llama2 model can be swapped with any other large language model by overwriting the custom LLM langchain class.


## Steps To run the project
1. Install the python requirements by using pip ``pip install -r requirements.txt``
2. Host the LLama2 model on colab and copy the ngrok URL.
3. Paste the Ngrok URL in the globals file ``MODEL_URL = NGROK_URL``
4. Run the project using ``streamlit run app.py`` 

## Colab file to host LLama2 on Colab GPU 
https://colab.research.google.com/drive/1Yxst2uOyWYVacXHA10WavO3L_ypg-8nZ?usp=sharing

## Workflow
<img src="https://github.com/bigdwarf43/Abstractive-Text-Summarization-Using-Llama2/assets/62785185/6ab9281b-6c89-41c9-a364-43122a635a41" width="500" height="800">

<img src="https://github.com/bigdwarf43/Abstractive-Text-Summarization-Using-Llama2/assets/62785185/3b062b53-d69a-4949-84f1-a64e582eefb9" width="500" height="500">

## Summarization example
![Summarizer-main](https://github.com/bigdwarf43/langchain-pdf-chat/assets/62785185/6f4bbf66-0f88-44c9-adf4-2bc0f26c6b37)

![AllChunkSummaries](https://github.com/bigdwarf43/langchain-pdf-chat/assets/62785185/12c182ef-6649-47d6-921e-5b934f3b1513)

![Summarizer-Summary](https://github.com/bigdwarf43/langchain-pdf-chat/assets/62785185/fcc12c0b-8b2a-4b45-9f9a-9ef3d0206f3e)


## RAG implementation example

![cloud1](https://github.com/bigdwarf43/langchain-pdf-chat/assets/62785185/8545ebd2-f705-4640-9400-9f48c5dc82f7)

![cloud2](https://github.com/bigdwarf43/langchain-pdf-chat/assets/62785185/d503ca2a-a6d1-44f7-9aa7-27fdb724bad2)



