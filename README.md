Our project allows users to input multiple PDFs to generate a concise summary that encompasses key points from all documents. 
Users can also engage in question-answering with the PDFs using the implemented Retrieval Augmented Generation workflow.

Our pipeline uses Langchain, Qdrant vector database and the LLama2 large language model.
The Llama2 model can be swapped with any other large language model by overwriting the custom LLM langchain class.


# Steps To run the project
1. Install the python requirements by using pip ``pip install -r requirements.txt``
2. Host the LLama2 model on colab and copy the ngrok URL.
3. Paste the Ngrok URL in the globals file ``MODEL_URL = NGROK_URL``
4. Run the project using ``streamlit run app.py`` 

## Colab file to host LLama2 on Colab GPU 
https://colab.research.google.com/drive/1Yxst2uOyWYVacXHA10WavO3L_ypg-8nZ?usp=sharing
