import nltk
import re
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest

# --- NLTK Resource Management (Run once on module load) ---
def _ensure_nltk_resources():
    """Checks and downloads necessary NLTK resources."""
    resources = ['stopwords', 'punkt', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            # Check for specific resource, e.g., stopwords.words('english')
            if resource == 'stopwords':
                stopwords.words('english')
            elif resource == 'punkt':
                sent_tokenize("Test.")
            elif resource == 'averaged_perceptron_tagger':
                nltk.pos_tag(word_tokenize("Test"))
        except LookupError:
            print(f"Downloading NLTK resource: {resource}...")
            nltk.download(resource)

_ensure_nltk_resources() 
# -----------------------------------------------------------

class TextRankSummarizer:
    """
    A Hybrid Extractive-Compressive Summarizer based on TextRank
    and deterministic Part-of-Speech (POS) based sentence pruning.
    
    The summary length is controlled by the percentage of the original total word count.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()

    def _clean_and_process_sentences(self, text):
        """Tokenizes text and cleans sentences for vectorization."""
        # Use NLTK's robust sentence tokenizer
        original_sentences = sent_tokenize(text)
        
        cleaned_sentences = []
        for s in original_sentences:
            # Remove non-alphabetic characters and lowercase
            s_clean = re.sub(r'[^a-zA-Z\s]', '', s.lower())
            words = s_clean.split()
            # Remove stop words
            s_processed = ' '.join([word for word in words if word not in self.stop_words])
            cleaned_sentences.append(s_processed)

        return original_sentences, cleaned_sentences

    def _compress_sentence(self, sentence):
        """
        Compresses a single sentence by removing non-essential POS tags.
        
        If an error occurs during POS tagging (e.g., missing resource), the
        original sentence words are preserved to prevent returning an empty string.
        """
        words = word_tokenize(sentence)
        
        try:
            # Part-of-Speech Tagging
            tagged_words = nltk.pos_tag(words)
        except Exception as e:
            # If POS tagging fails, use original words without compression
            print(f"Warning: POS tagging failed for sentence. Returning original words. Error: {e}")
            compressed_words = words
            
            # If tagging failed, skip to the joining section
            compressed_text = ' '.join(compressed_words)
            compressed_text = re.sub(r'\s([,\.\?!:])', r'\1', compressed_text)
            return compressed_text.capitalize()
            
        # Define the set of essential tags (core SVO)
        essential_tags = {'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                          'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', # Verbs
                          'PRP', 'DT', 'CD', 'IN', 'CC'} # Pronouns, Determiners, Numbers, Prepositions, Conjunctions

        compressed_words = []
        for word, tag in tagged_words:
            # Simple check: keep if the tag is in the essential set
            if tag in essential_tags or tag.startswith('NN') or tag.startswith('VB'):
                 compressed_words.append(word)
        
        # FIX: If compression removed ALL words (but there were input words),
        # return original words to avoid empty string.
        if not compressed_words and words:
             compressed_words = [word for word, tag in tagged_words if word.isalnum()]
             if not compressed_words: # Final safeguard
                 compressed_words = words

        # Simple join. A more advanced method would re-insert proper punctuation.
        compressed_text = ' '.join(compressed_words)
        
        # Simple fix for punctuation spacing
        compressed_text = re.sub(r'\s([,\.\?!:])', r'\1', compressed_text)
        
        return compressed_text.capitalize() # Start with a capital letter

    def _handle_single_sentence_case(self, original_sentences, cleaned_sentences, target_word_count):
        """
        Handle the special case when there's only one sentence.
        Applies compression and returns the result.
        """
        if not original_sentences:
            return "Cannot summarize empty content."
            
        single_sentence = original_sentences[0]
        compressed = self._compress_sentence(single_sentence)
        compressed_words = compressed.split()
        
        # If the compressed version is within target or we have only one sentence,
        # return the compressed version
        if len(compressed_words) <= target_word_count or len(original_sentences) == 1:
            return compressed
        else:
            # If compressed is still too long, truncate to target word count
            return ' '.join(compressed_words[:target_word_count])

    def summarize(self, input_text, summary_percentage=0.7):
        """
        Generates the compressed summary.
        
        The summary length is determined by a percentage of the original word count.

        Args:
            input_text (str): The document to summarize.
            summary_percentage (float): Percentage of original total word count to target.
        """
        original_sentences, cleaned_sentences = self._clean_and_process_sentences(input_text)

        if not original_sentences or all(not s for s in cleaned_sentences):
            return "Cannot summarize empty or non-text content."

        # Calculate total original words for percentage target
        total_original_words = sum(len(word_tokenize(s)) for s in original_sentences)
        target_word_count = int(total_original_words * summary_percentage)
        
        if target_word_count == 0:
            # If the text is short and target is small, ensure at least one word is targeted
            target_word_count = 1

        # --- SPECIAL CASE: Single sentence ---
        if len(original_sentences) == 1:
            return self._handle_single_sentence_case(original_sentences, cleaned_sentences, target_word_count)

        # --- MULTIPLE SENTENCES: Normal TextRank processing ---
        
        try:
            sentence_vectors = self.vectorizer.fit_transform(cleaned_sentences)
        except ValueError:
            return "The document contains no significant words after cleaning."

        # Building the Graph via Cosine Similarity
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Create graph and calculate PageRank scores
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph) 

        # Map scores to original indices
        ranked_sentences = {i: scores[i] for i in range(len(original_sentences))}
        
        # Sort all sentences by score in descending order
        sorted_ranks = sorted(ranked_sentences.items(), 
                              key=lambda item: item[1], 
                              reverse=True)
                                             
        # --- WORD-BASED SELECTION, COMPRESSION & RECONSTRUCTION ---
        
        selected_sentences_by_index = {}
        current_word_count = 0
        
        # Iterate through sentences from most important to least important
        for index, score in sorted_ranks:
            original_sentence = original_sentences[index]
            
            # 1. Compress the sentence deterministically
            compressed_sentence = self._compress_sentence(original_sentence)
            compressed_length = len(compressed_sentence.split())
            
            # 2. Check if adding the sentence exceeds the target word count
            if current_word_count + compressed_length <= target_word_count:
                selected_sentences_by_index[index] = compressed_sentence
                current_word_count += compressed_length
            else:
                # If we go over the limit, stop adding sentences
                break
                
        # If no sentences were selected (all were too long), pick the top one
        if not selected_sentences_by_index and sorted_ranks:
            top_index = sorted_ranks[0][0]
            compressed = self._compress_sentence(original_sentences[top_index])
            compressed_words = compressed.split()
            # Return truncated version if needed
            if len(compressed_words) > target_word_count:
                selected_sentences_by_index[top_index] = ' '.join(compressed_words[:target_word_count])
            else:
                selected_sentences_by_index[top_index] = compressed
                
        # 3. Reconstruct the summary in original order
        final_summary_parts = []
        
        # Sort the selected sentences by their original index (key)
        for index in sorted(selected_sentences_by_index.keys()):
            final_summary_parts.append(selected_sentences_by_index[index])
            
        return " ".join(final_summary_parts)
