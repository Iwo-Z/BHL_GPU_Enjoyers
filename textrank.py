import nltk
import re
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest

# --- NLTK Resource Management (Run once on module load) ---
def _ensure_nltk_resources():
    """Checks and downloads necessary NLTK resources."""
    resources = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
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
    """

    def __init__(self, damping_factor=0.85):
        self.damping_factor = damping_factor
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
        
        The compression rules are deterministic and aim to keep the core
        Subject-Verb-Object (SVO) structure intact.
        
        Tags to keep (core structure):
        NN*, VB*, CD (Nouns, Verbs, Numbers)
        PRP, DT (Pronouns, Determiners like 'the')
        IN (Prepositions) - often needed to link core phrases
        
        Tags to prune (non-essential modifiers/adverbials):
        JJ*, RB* (Adjectives, Adverbs) - removing these simplifies
        W* (Wh-words) - often used in non-restrictive clauses (which are complex to handle fully without a full parser)
        """
        words = word_tokenize(sentence)
        
        # Part-of-Speech Tagging
        tagged_words = nltk.pos_tag(words)

        # Define the set of essential tags (core SVO)
        essential_tags = {'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                          'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', # Verbs
                          'PRP', 'DT', 'CD', 'IN', 'CC'} # Pronouns, Determiners, Numbers, Prepositions, Conjunctions

        compressed_words = []
        for word, tag in tagged_words:
            # Simple check: keep if the tag is in the essential set
            if tag in essential_tags or tag.startswith('NN') or tag.startswith('VB'):
                 compressed_words.append(word)
            # Exception: Keep common structural words like 'is', 'are', 'was' if they are verbs
            # This is largely covered by 'VB*' tags, but ensures we don't over-prune linking verbs.
        
        # Simple join. A more advanced method would re-insert proper punctuation.
        compressed_text = ' '.join(compressed_words)
        
        # Simple fix for punctuation spacing
        compressed_text = re.sub(r'\s([,\.\?!:])', r'\1', compressed_text)
        
        return compressed_text.capitalize() # Start with a capital letter

    def summarize(self, input_text, summary_percentage=0.3):
        """
        Generates the compressed summary.

        Args:
            input_text (str): The document to summarize.
            summary_percentage (float): Percentage of original sentences to select.
        """
        original_sentences, cleaned_sentences = self._clean_and_process_sentences(input_text)

        if not original_sentences or all(not s for s in cleaned_sentences):
            return "Cannot summarize empty or non-text content."

        num_sentences_to_select = max(1, int(len(original_sentences) * summary_percentage))

        # --- PHASE 1 & 2: SCORING & SELECTION (TextRank) ---
        
        try:
            sentence_vectors = self.vectorizer.fit_transform(cleaned_sentences)
        except ValueError:
            return "The document contains no significant words after cleaning."

        # Building the Graph via Cosine Similarity
        similarity_matrix = cosine_similarity(sentence_vectors)
        

        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, alpha=self.damping_factor) 

        # Selection of top N sentences
        ranked_sentences = {i: scores[i] for i in range(len(original_sentences))}
        top_indices_with_score = nlargest(num_sentences_to_select, 
                                          ranked_sentences.items(), 
                                          key=lambda item: item[1])
                                             
        top_indices = {index for index, score in top_indices_with_score}
        
        # --- PHASE 3 & 4: COMPRESSION & RECONSTRUCTION ---
        
        final_summary = []
        
        for i, original_sentence in enumerate(original_sentences):
            if i in top_indices:
                # Compression step is applied only to the selected sentences
                compressed_sentence = self._compress_sentence(original_sentence)
                final_summary.append(compressed_sentence)
                
        return " ".join(final_summary)

