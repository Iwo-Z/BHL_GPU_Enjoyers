import nltk
import re
import networkx as nx
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest

# Pobieranie zasobów NLTK na poziomie modułu
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

class TextRankSummarizer:
    """
    Klasa implementująca algorytm ekstraktywnej sumaryzacji TextRank (PageRank)
    oparty na podobieństwie zdań TF-IDF.
    """

    def _preprocess_text(self, text):
        """
        Prywatna metoda do wstępnego przetwarzania tekstu.
        """
        # Tokenizacja na zdania
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        english_stop_words = stopwords.words('english')
        
        cleaned_sentences = []
        for s in sentences:
            s_clean = re.sub(r'[^a-zA-Z\s]', '', s.lower())
            words = s_clean.split()
            s_processed = ' '.join([word for word in words if word not in english_stop_words])
            cleaned_sentences.append(s_processed)

        return sentences, cleaned_sentences

    def summarize(self, input_text, summary_percentage=0.7):
        """
        Sumaryzuje podany tekst, zwracając streszczenie w oryginalnej kolejności zdań.

        Args:
            input_text (str): Tekst do streszczenia.
            summary_percentage (float): Procent oryginalnego tekstu do zachowania.
            
        Returns:
            str: Wygenerowane streszczenie.
        """
        original_sentences, cleaned_sentences = self._preprocess_text(input_text)

        if not original_sentences or all(not s for s in cleaned_sentences):
            return ""

        num_sentences_to_select = max(1, int(len(original_sentences) * summary_percentage))

        vectorizer = TfidfVectorizer()
        
        sentence_vectors = vectorizer.fit_transform(cleaned_sentences)

        similarity_matrix = cosine_similarity(sentence_vectors)

        graph = nx.from_numpy_array(similarity_matrix)

        scores = nx.pagerank(graph)
        
        ranked_sentences = {original_sentences[i]: scores[i] for i in range(len(original_sentences))}

        best_sentences_with_score = nlargest(num_sentences_to_select, 
                                            ranked_sentences.items(), 
                                            key=lambda item: item[1])
                                            
        best_sentences = [sentence for sentence, score in best_sentences_with_score]

        final_summary = []
        
        for original_sentence in original_sentences:
            if original_sentence in best_sentences:
                final_summary.append(original_sentence)
                
        return " ".join(final_summary)
    
    
    
text = "The hillside laboratory buzzed with a faint metallic hum as scattered notebooks fluttered in the draft from an unseen vent. A rusty clock on the wall insisted it was still morning, though the sky outside had already slipped into a muted violet haze. In the corner, a glass tank filled with drifting blue spores pulsed softly, responding to some pattern no one had yet decoded. Researchers whispered theories about resonance fields and forgotten mapping algorithms, but none dared to touch the console after last week’s incident. Meanwhile, the corridor lights flickered at irregular intervals, as if keeping time with a melody only the building could hear."



print(len(text))

print(len(TextRankSummarizer().summarize(text)))