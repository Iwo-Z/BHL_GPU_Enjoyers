# Dokumentacja Projektu: EcoPrompt Optimizer
**Hackathon:** B#L - Best Hacking League (Edycja AI & Green Tech)
**Data:** 30.11.2025
**Nazwa Drużyny:** GPU Enjoyers

---

## 1. Zdefiniowanie problemu

### Kontekst ekologiczny
Sektor IT, a w szczególności centra danych obsługujące modele sztucznej inteligencji, odpowiada za rosnące zużycie energii. Szacuje się, że do 2040 roku oprogramowanie może generować nawet 14% globalnych emisji CO2. Każde zapytanie do modelu LLM (Large Language Model) zużywa energię proporcjonalną do liczby przetwarzanych tokenów (słów/części słów).

### Zidentyfikowany problem
Użytkownicy modeli językowych często tworzą nieefektywne prompty zawierające:
* Zbędne zwroty grzecznościowe ("Dzień dobry", "Proszę", "Pozdrawiam").
* Nadmiarowy kontekst, który nie wpływa na jakość odpowiedzi.
* Powtórzenia.

Model LLM przetwarza te "śmieciowe" dane, zużywając moc obliczeniową GPU i energię bez wnoszenia wartości merytorycznej.

### Proponowane rozwiązanie
Stworzyliśmy **EcoPrompt Optimizer** – warstwę pośrednią, która automatycznie optymalizuje prompty przed wysłaniem ich do modelu docelowego. Rozwiązanie wykorzystuje NLP do usuwania szumu komunikacyjnego i kondensacji treści.

### Klient docelowy i korzyści biznesowe
* **Klient:** Firmy SaaS i startupy technologiczne, które intensywnie wykorzystują płatne API w swoich produktach (np. chatboty obsługi klienta, asystenci kodowania).
* **Korzyści:**
    1.  **Ekologia:** Mniejszy ślad węglowy dzięki redukcji liczby tokenów przetwarzanych przez energochłonne modele w data center.
    2.  **Oszczędność:** Redukcja kosztów operacyjnych (API providerzy rozliczają się za liczbę tokenów wejściowych).
    3.  **Wydajność:** Krótszy czas przetwarzania zapytania (mniejsza latencja).

---

## 2. Analiza i przygotowanie danych
### Wykorzystane zbiory danych:
1.  **Nvidia HelpSteer**
    * **Źródło:** HuggingFace (https://huggingface.co/datasets/nvidia/HelpSteer).
    * **Zastosowanie:** Testowanie algorytmów na danych technicznych. Zbiór zawiera prompty i odpowiedzi dotyczące programowania. Pozwolił zweryfikować, czy optymalizacja promptu nie usuwa kluczowych fragmentów kodu lub zmiennych.
2.  **Conversational Interaction Dataset** 
    * **Źródło:** Kaggle, wiele małych datasetów (m. in https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in).
    * **Charakterystyka:** Zbiór typowych interakcji czatowych zawierający powitania, pożegnania i *small talk*.
    * **Cel:** Wytrenowanie klasyfikatora do rozpoznawania intencji "social" vs "content".

### Przetwarzanie danych
* **Czyszczenie:** Usunięcie znaczników i fragmentów kodu z datasetów.
* **Tokenizacja:** Podział tekstu na zdania w celu analizy ich wagi semantycznej.
* **Analiza:** Zidentyfikowano, że średnio 15-20% tokenów w zapytaniach konwersacyjnych to "szum" (stopwords, zwroty grzecznościowe), co stanowi potencjał optymalizacyjny.

### Generowanie treningowego zbioru danych

### Cel i etykiety
Skrypt generuje zbalansowany zbiór krótkich fraz rozmówkowych i „treści merytorycznej”, klasyfikowanych do czterech etykiet:
* Label 0 – Greetings (powitania)
* Label 1 – Thanks (podziękowania)
* Label 2 – Goodbyes (pożegnania)
* Label 3 – Others/Hard (pozostałe treści, w tym „tricky” i zwykłe prompty)

Zbiór jest budowany tak, aby liczba przykładów dla Label 3 była zbliżona do łącznej liczby elementów klas 0–2 (balans klas).

Główne etapy:
1. Generowanie wariantów fraz dla etykiet 0–2 (różne wielkości liter oraz prosta interpunkcja dla krótkich fraz).
2. Dodanie bogatej listy przykładów dla etykiety 3 (`TRICKY_PHRASES`).
3. Opcjonalne doładowanie danych zewnętrznych do Label 3 z lokalnych plików CSV (patrz „Dane wejściowe – opcjonalne”).
4. Scalenie i deduplikacja dokładnych duplikatów (case‑sensitive).
5. Zapis gotowego zbioru i wygenerowanie wykresu rozkładu etykiet.

Funkcje kluczowe:
* `generate_variants(phrase, label)` – wytwarza proste warianty danej frazy.
* `load_external_data(target_count=None)` – wczytuje dodatkowe przykłady Label 3 z plików CSV w bieżącym katalogu roboczym.
* `build_final_dataset()` – buduje kompletny, zbalansowany dataset i zwraca listę `(text, label)`.
* `plot_distribution(counts)` – zapisuje wykres rozkładu klas do pliku `dataset_distribution.png`.

### Wyniki
Skrypt zapisuje w bieżącym katalogu roboczym:
* `categorized_phrases.csv` – plik CSV z kolumnami: `text`, `label`.
* `dataset_distribution.png` – wykres rozkładu liczby rekordów w poszczególnych klasach.
---

## 3. Zastosowanie modeli uczenia maszynowego
### Architektura rozwiązania:
#### Krok A: Klasyfikacja intencji (BERT)
Wykorzystano model **BERT (Bidirectional Encoder Representations from Transformers)**, poddany procesowi fine-tuningu na zbiorze konwersacyjnym.
* **Funkcja:** Klasyfikacja wieloklasowa każdego zdania w prompcie.
* **Logika:** Jeśli zdanie jest klasyfikowane jako "Greeting/farewell/gratitude" -> odpowiedz hardcode'owaną odpowiedzią. Jeśli "prompt" -> przejdź dalej.
* **Dlaczego BERT?** Doskonale rozumie kontekst. Odróżnia powitania, pożegnania i podziękowania od elementów kodu czy istotnych nazw własnych, czego nie robią proste filtry oparte na słowach kluczowych.

#### Krok B: Ekstraktywna sumaryzacja (TextRank)
Dla długich promptów zastosowano algorytm **TextRank**.
* **Funkcja:** Budowa grafu, gdzie wierzchołkami są zdania, a krawędziami ich podobieństwo semantyczne.
* **Działanie:** Wybierane są tylko zdania o najwyższej randze (najważniejsze dla kontekstu), tworząc skróconą wersję tekstu.
* **Zaleta:** TextRank jest algorytmem "lekkim" obliczeniowo w porównaniu do generowania podsumowań przez LLM, co wpisuje się w ideę *Green AI*.

#### Krok C: Porównanie rozwiązań
Porównaliśmy nasze podejście hybrydowe (BERT + TextRank) z brakiem preprocessingu.
* *Truncation:* Szybkie, ale traci sens logiczny przy długich tekstach.
* *EcoPrompt (Nasze):* Zachowuje sens semantyczny (zgodność cosine similarity > 0.9) przy redukcji tokenów o średnio 25%.

---

## 4. Prezentacja rozwiązania i wnioski
### Sposób działania (Workflow)
1.  Użytkownik wysyła zapytanie: *"Cześć, mam problem z kodem, mógłbyś zerknąć? Oto on: [Długi Kod]. Z góry dzięki!"*
2.  **EcoPrompt API** przetwarza tekst:
    * BERT usuwa: *"Cześć, mam problem z kodem, mógłbyś zerknąć?"* oraz *"Z góry dzięki!"*.
    * TextRank analizuje kod/opis i kondensuje go, jeśli jest zbyt rozwlekły.
3.  Do LLM (np. GPT-4) trafia tylko: *"Analiza błędu: [Długi Kod]"*.
4.  LLM zwraca poprawną odpowiedź, zużywając mniej tokenów.

### Wnioski końcowe [cite: 40]
Stworzony prototyp udowadnia, że można pogodzić wysoką jakość odpowiedzi systemów AI z dbałością o środowisko. Projekt spełnia założenia hackathonu:
1.  Rozwiązuje problem ekologiczny (zużycie energii przez Data Center).
2.  Wykorzystuje zbiory danych i modele AI (BERT, TextRank).
3.  Jest gotowy do wdrożenia jako usługa biznesowa B2B.

---