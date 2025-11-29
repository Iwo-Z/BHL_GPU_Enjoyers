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

    def __init__(self, damping_factor=0.8):
        """
        Inicjalizuje sumaryzator.
        Args:
            damping_factor (float): Współczynnik tłumienia
        """
        self.damping_factor = damping_factor
        self.stop_words = stopwords.words('english')
        self.vectorizer = TfidfVectorizer()

    def _preprocess_text(self, text):
        """
        Prywatna metoda do wstępnego przetwarzania tekstu.
        """
        # Tokenizacja na zdania
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        cleaned_sentences = []
        for s in sentences:
            s_clean = re.sub(r'[^a-zA-Z\s]', '', s.lower())
            words = s_clean.split()
            s_processed = ' '.join([word for word in words if word not in self.stop_words])
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

        try:
             # Używamy fit_transform na nowo dla każdego tekstu, aby stworzyć nowy słownik słów
            sentence_vectors = self.vectorizer.fit_transform(cleaned_sentences)
        except ValueError:
            return ""

        # Macierz Podobieństwa
        # Obliczanie podobieństwa cosinusowego
        similarity_matrix = cosine_similarity(sentence_vectors)
        

        # TextRank
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Obliczanie rankingu zdań
        scores = nx.pagerank(graph) 
        
        ranked_sentences = {original_sentences[i]: scores[i] for i in range(len(original_sentences))}

        # N zdań o najwyższych wynikach
        best_sentences_with_score = nlargest(num_sentences_to_select, 
                                             ranked_sentences.items(), 
                                             key=lambda item: item[1])
                                             
        best_sentences_set = {sentence for sentence, _ in best_sentences_with_score}
        
        final_summary = [s for s in original_sentences if s in best_sentences_set]
                
        return " ".join(final_summary)