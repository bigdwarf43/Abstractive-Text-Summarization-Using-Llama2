# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import streamlit as st
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

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
        


        
        # Find the optimal number of clusters using the elbow method
        distortions = []
        cluster_range = range(3, 15)  # You can adjust the range based on your data
        for k in cluster_range:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
            kmeans.fit(vectors)
            distortions.append(kmeans.inertia_)


        # Plot the distortions to find the elbow point
        plt.figure(figsize=(8, 6))
        plt.plot(cluster_range, distortions, marker='o')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.savefig('foo2.png')

        # Choose the number of clusters, this can be adjusted based on the text's content.
        elbow_point = None
        for i in range(1, len(distortions) - 1):
            if distortions[i] < distortions[i - 1] and distortions[i] < distortions[i + 1]:
                elbow_point = i + 1

        if elbow_point is not None:
            optimal_num_clusters = cluster_range[elbow_point - 1]
            print(f"Optimal number of clusters based on the elbow method: {optimal_num_clusters}")
        else:
            print("No clear elbow point found, please manually select the optimal number of clusters.")
            optimal_num_clusters = 10


        #perform k-means clustering
        kmeans = MiniBatchKMeans(n_clusters=optimal_num_clusters, random_state=42)
        kmeans.fit(vectors)


        #Write the embeddings to embeddings.tsv

        with open('embeddings.tsv', 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')

            for vect in vectors:
                writer.writerow(vect)


        # Get the cluster labels for each data point
        cluster_labels = kmeans.labels_

        # Create a dictionary to store the cluster labels for each data point
        cluster_dict = {i: cluster_labels[i] for i in range(len(cluster_labels))}

        print("Clusters:\n")
        print(cluster_dict)
        

        # Find the closest point to each cluster center
        closest_indices = np.argmin(kmeans.transform(vectors), axis=0)

        # Sort the closest indices
        selected_indices = np.sort(closest_indices) 

        print("Selected cluster centers:\n")
        print(selected_indices)
        print("The cluster length: "+str(len(selected_indices)))

        map_prompt = """
        [INST]
        <<SYS>>
        Write a summary of the following text.
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
        The following is set of summaries:
        {text}
        Take these and distil it into a final consolidated summary with title(mandatory) in bold with important key points . 
        SUMMARY: 
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])


        reduce_chain = load_summarize_chain(llm=self.llm,
                                            chain_type="stuff",
                                            prompt=combine_prompt_template
                                        )
        
        output = reduce_chain.run([summaries])


        # map_template = """The following is a set of documents
        # {docs}
        # Based on this list of docs, summarised into meaningful
        # Helpful Answer:"""

        # map_prompt = PromptTemplate.from_template(map_template)
        # map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # reduce_template = """The following is set of summaries:
        # {doc_summaries}
        # Take these and distil it into a final consolidated summary with title(mandatory) in bold with important key points . 
        # Helpful Answer:"""

        # reduce_prompt = PromptTemplate.from_template(reduce_template)
        # reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)


        # combine_documents_chain = StuffDocumentsChain(
        #     llm_chain=reduce_chain, document_variable_name="doc_summaries"
        # )
        # reduce_documents_chain = ReduceDocumentsChain(
        #     combine_documents_chain=combine_documents_chain,
        #     collapse_documents_chain=combine_documents_chain,
        #     token_max=5000,
        # )

        # map_reduce_chain = MapReduceDocumentsChain(
        # llm_chain=map_chain,
        # reduce_documents_chain=reduce_documents_chain,
        # document_variable_name="docs",
        # return_intermediate_steps=False,
        # )

        # output = map_reduce_chain.run(selected_docs)
        # summary_list = []

        return output, summary_list, selected_docs
        






