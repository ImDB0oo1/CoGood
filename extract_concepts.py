import spacy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class ExtractConcepts:
    def __init__(self, method=None):
        """
        Initialize the ExtractConcepts class with available methods.

        Parameters:
        - methods (list): List of concept extraction methods to use.
        """
        self.method = method
        # Initialize a pre-trained BERT model for sentence embeddings
        self.model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model

    def extract_embeddings(self, documents):
        return self.model.encode(documents)

    def extract_concept(self, ID_documents, OOD_documents=None, num_clusters=None, top_n=None):
        """
        Extract concepts from a given text based on the specified methods.

        Parameters:
        - ID_documents (list): List of ID documents.
        - OOD_documents (list): List of OOD documents (optional).
        - num_clusters (int): Number of clusters for KMeans.
        - top_n (int): Number of top sentences to extract.

        Returns:
        - Extracted concepts based on the method.
        """
        if self.method == "sentence":
            return self.extract_concept_sentences(ID_documents, OOD_documents, top_n)
        if self.method == "kmeans":
            return self.extract_concept_kmeans(ID_documents, num_clusters)

    def extract_concept_kmeans(self, ID_documents, num_clusters):
        """
        Extract topic clusters using KMeans.
        """
        embeddings = self.extract_embeddings(ID_documents)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_
        return cluster_centers

    def extract_concept_sentences(self, ID_documents, OOD_documents, top_n):
        """
        Extract topic sentences by calculating similarity scores.
        """
        ID_sentences = self.tokenize_documents(ID_documents)
        ID_sentences_embeddings = self.extract_embeddings(ID_sentences)
        ID_embeddings = self.extract_embeddings(ID_documents)
        OOD_embeddings = self.extract_embeddings(OOD_documents)

        return self.calculate_scores(
            ID_sentences, ID_sentences_embeddings, ID_embeddings, OOD_embeddings, top_n
        )

    @staticmethod
    def calculate_scores(ID_sentences, ID_sentences_embeddings, ID_embeddings, OOD_embeddings, top_n):
        """
        Calculate scores for sentences based on similarity to ID and OOD documents.
        """
        # Calculate cosine similarities
        sim_to_ID_docs = cosine_similarity(ID_sentences_embeddings, ID_embeddings)
        sim_to_OOD_sentences = cosine_similarity(ID_sentences_embeddings, OOD_embeddings)

        # Average cosine similarities
        avg_sim_to_ID_docs = sim_to_ID_docs.mean(axis=1)
        avg_sim_to_OOD_sentences = sim_to_OOD_sentences.mean(axis=1)

        # Calculate scores
        scores = avg_sim_to_ID_docs - avg_sim_to_OOD_sentences

        # Get indices of the top N sentences with highest scores
        top_indices = np.argsort(scores)[-top_n:]
        top_sentences_embeddings = ID_sentences_embeddings[top_indices]
        top_sentences = [ID_sentences[i] for i in top_indices]

        return top_sentences, top_sentences_embeddings

    def tokenize_documents(self, documents):
        """
        Tokenize multiple documents into sentences using spaCy and multiprocessing.
        """
        with ProcessPoolExecutor() as executor:
            sentences = list(executor.map(self.split_into_sentences, documents))

        # Flatten the list of sentences
        return [sentence for sublist in sentences for sentence in sublist]

    def split_into_sentences(self, doc):
        """
        Tokenize a single document into sentences using spaCy.
        """
        doc_nlp = self.nlp(doc)
        return [sent.text.strip() for sent in doc_nlp.sents]
