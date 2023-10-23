# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate

import streamlit as st

import numpy as np
from sklearn.cluster import KMeans

import csv

class summarizer():

    def __init__(self, llm, text) -> None:
        self.llm = llm
        self.text = text

    def summarize(self):

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=500, chunk_overlap=200)

        chunks = text_splitter.create_documents([self.text])

        print("Number of Chunks: "+str(len(chunks)))

        embeddings = GPT4AllEmbeddings()

        #create embeddings
        vectors = embeddings.embed_documents([x.page_content for x in chunks])
        
        # 'embeddings' is a list or array of 538-dimensional embeddings
        print("Vector dimensions:"+ str(len(vectors[0])))
        print("Vector dimensions:"+ str(len(vectors[1])))


        # Choose the number of clusters, this can be adjusted based on the text's content.
        num_clusters = 7

        #perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)


        #Write the embeddings to embeddings.tsv

        with open('embeddings.tsv', 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')

            for vect in vectors:
                writer.writerow(vect)


        print("Clusters:\n")
        dicts = {i:kmeans.labels_[i] for i in range(0, len(kmeans.labels_))}
        print(dicts)
        
        # Create an empty list that will hold your closest points
        closest_indices = []

        # Loop through the number of clusters you have
        for i in range(num_clusters):
            
            # Get the list of distances from that particular cluster center
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            
            # Find the list position of the closest one (using argmin to find the smallest distance)
            closest_index = np.argmin(distances)
            
            # Append that position to your closest indices list
            closest_indices.append(closest_index)

        
        selected_indices = sorted(closest_indices)

        print("Selected cluster centers:\n")
        print(selected_indices)

        map_prompt = """
        [INST]
        <<SYS>>
        Write a summary of the following text delimited by triple backticks.
        Return your response which covers the key points of the text.
        <</SYS>>

        ```{text}```
        SUMMARY: [/INST]
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

        map_chain = load_summarize_chain(llm=self.llm,
                             chain_type="stuff",
                             prompt=map_prompt_template)
        
        selected_docs = [chunks[doc] for doc in selected_indices]

        # Make an empty list to hold your summaries
        summary_list = []

        # Loop through a range of the lenght of your selected docs
        for i, doc in enumerate(selected_docs):

            # Go get a summary of the chunk
            chunk_summary = map_chain.run([doc])
            
            # Append that summary to your list
            summary_list.append(chunk_summary)
            
            print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")

        summaries = "\n".join(summary_list)

        summaries = Document(page_content=summaries)

        print(summaries)

        combine_prompt = """
        [INST]
        <<SYS>>
        You will be given a series of summaries from a research paper. The summaries will be enclosed in triple backticks (```)
        Your goal is to use these summaries as context and give a concise summary of what is given in the research paper. Output the summary in paragraphs.
        The reader should be able to grasp what is mentioned in the research paper.
        <</SYS>>

        ```{text}```
        VERBOSE SUMMARY: [/INST]
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])


        reduce_chain = load_summarize_chain(llm=self.llm,
                                            chain_type="stuff",
                                            prompt=combine_prompt_template
                                        )
        
        output = reduce_chain.run([summaries])
        return output, summary_list, selected_docs
        






