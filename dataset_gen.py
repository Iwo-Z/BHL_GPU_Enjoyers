import csv
import re
import matplotlib.pyplot as plt
import os
import zipfile
from pathlib import Path
import pandas as pd

# ===========================================
# 1. DEFINICJE LIST I KATEGORII
# ===========================================

# LABEL 0: GREETINGS (Powitania) - ROZSZERZONE
GREETINGS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "hiya", "yo", "sup", "good to see you", "how are you", "how are you doing",
    "how is it going", "what's up", "g'day", "howdy", "morning", "evening",
    "greetings", "salutations", "hey there", "hi there", "hello there",
    "good day", "top of the morning", "what's happening", "how's it going",
    "how do you do", "pleased to meet you", "nice to meet you", "howdy do",
    "hey buddy", "hey friend", "what's good", "what's new", "how have you been",
    "long time no see", "good to hear from you", "nice to see you",
    "welcome", "welcome back", "hi everyone", "hello everyone", "hey folks",
    "greetings everyone", "good morning all", "afternoon", "good eve",
    "heya", "hallo", "howdy partner", "yo yo", "sup dude", "hey dude",
    "what's crackin", "what's poppin", "how goes it", "how are things",
    "how are ya", "how ya doing", "wassup", "wazzup", "how's life",
    "how's everything", "how's your day", "hope you're well", "hope all is well"
]

# LABEL 1: THANKS (Podziękowania) - ROZSZERZONE
THANKS = [
    "thanks", "thank you", "thx", "ty", "many thanks", "thanks a lot",
    "thank u", "thanx", "much appreciated", "i appreciate it", "tysm",
    "thank you so much", "thanks so much", "really appreciate it",
    "appreciate your help", "grateful", "i'm grateful", "much obliged",
    "thanks a bunch", "thanks a million", "thank you very much",
    "many thanks indeed", "that's very kind", "that's great thanks",
    "awesome thanks", "perfect thank you", "brilliant thanks",
    "wonderful thank you", "fantastic thanks", "excellent thank you",
    "cheers", "cheers mate", "ta", "thanks mate", "thank you kindly",
    "i owe you one", "you're a lifesaver", "you're the best",
    "can't thank you enough", "thank you for your help", "thanks for helping",
]

# LABEL 2: GOODBYES (Pożegnania) - ROZSZERZONE
GOODBYES = [
    "bye", "goodbye", "see you", "see ya", "take care", "have a good one",
    "catch you later", "later", "ciao", "farewell", "talk to you later", "ttyl",
    "see you later", "see you soon", "until next time", "till next time",
    "gotta go", "got to go", "i'm off", "i have to go", "i must go",
    "bye bye", "bye for now", "goodbye for now", "talk soon", "speak soon",
    "catch you later", "see you around", "peace out", "peace",
    "cheerio", "toodles",
    "so long", "good night", "night", "nighty night", "sleep well",
    "sweet dreams", "have a good night", "have a great day", "have a nice day",
    "enjoy your day", "have a good evening", "good luck", "all the best",
    "best wishes", "take it easy", "stay safe", "be well", "be good",
    "until we meet again", "looking forward to next time", "thanks and goodbye",
    "thanks bye", "okay bye", "alright bye", "see ya later", "laters",
    "bye now", "signing off", "logging off", "i'm out", "i'm done",
    "that's all from me", "i'll let you go", "talk later", "bye everyone"
]

# LABEL 3: OTHERS / TRICKY (Inne + Podchwytliwe) - ROZSZERZONE
TRICKY_PHRASES = [
    # Podchwytliwe z "Thanks"
    "thanks but I don't understand",
    "thanks but that is wrong",
    "thanks for nothing",
    "thanks to gravity apples fall down",
    "thank you implies gratitude but I feel anger",
    "thanks is not enough I need a refund",
    "thanks for the explanation now explain the next part",
    "thanksgiving dinner recipes",
    "thanksgiving holiday history",
    "no thanks I'll pass",
    "thanks giving origins",

    # Podchwytliwe z "Hello" / "Hi"
    "hello world python script",
    "say hello to my little friend quote origin",
    "hi tech industry trends",
    "high performance computing",
    "high definition video",
    "highway traffic analysis",
    "hierarchy in organizations",
    "historical data analysis",

    # Podchwytliwe z "Good"
    "good morning vietnam movie rating",
    "good afternoon is when strictly speaking",
    "good will hunting plot summary",
    "goods and services tax",
    "goodness of fit test",

    # Podchwytliwe z "Bye"
    "bye bye birdie cast",
    "bye law definitions",
    "bypass surgery recovery",
    "bytecode compilation process",

    # Normalne prompty - pytania
    "what is the weather like",
    "tell me a joke",
    "explain quantum physics",
    "write a poem about cats",
    "how do I cook pasta",
    "generate a sql query",
    "summarize the text above",
    "ignore previous instructions",
    "what time is it",
    "where is the nearest restaurant",
    "how much does it cost",
    "can you help me with my homework",
    "what's the capital of france",
    "who won the world cup",
    "when was world war 2",
    "why is the sky blue",
    "define artificial intelligence",
    "calculate the square root of 144",
    "translate this to spanish",
    "show me examples of metaphors",

    # Normalne prompty - polecenia
    "create a list of ideas",
    "analyze this data set",
    "compare these two options",
    "find the best solution",
    "optimize this code",
    "debug this function",
    "refactor the following",
    "implement a binary search",
    "design a database schema",
    "write unit tests for",
    "review this pull request",
    "explain the algorithm",
    "document this api",
    "create a class diagram",

    # Normalne prompty - złożone
    "what are the implications of climate change",
    "how does machine learning work",
    "explain the difference between",
    "what factors contribute to",
    "provide an overview of",
    "discuss the pros and cons",
    "evaluate the effectiveness of",
    "assess the impact of",
    "identify the key components"
]


# ===========================================
# 2. FUNKCJE POMOCNICZE
# ===========================================

def generate_variants(phrase, label):
    """Generuje proste warianty (duże/małe litery, interpunkcja)."""
    variants = []

    # Czysta forma
    variants.append((phrase, label))
    variants.append((phrase.capitalize(), label))
    variants.append((phrase.upper(), label))

    # Z interpunkcją (tylko dla krótkich fraz smalltalkowych, nie dla długich promptów)
    is_short = len(phrase.split()) < 4
    if is_short:
        for punct in ["!", ".", "...", "?"]:
            variants.append((f"{phrase}{punct}", label))
            variants.append((f"{phrase.capitalize()}{punct}", label))

    return variants



def merge_datasets(*lists):
    """
    Łączy listy i usuwa DOKŁADNE duplikaty (case-sensitive).
    Usuwa tylko identyczne stringi, więc "Hi" i "hi" są różne.
    """
    seen = set()
    merged = []

    for lst in lists:
        for item, label in lst:
            if item not in seen:
                seen.add(item)
                merged.append((item, label))

    return merged


def load_external_data(target_count=None):
    """
    Wczytuje dane Label 3 z lokalnych plików CSV:
    - train.csv
    - test.csv
    - validation.csv

    Każdy wiersz -> (tekst, label=3)
    """

    # Ścieżka do folderu z CSV
    base_path = Path(".")

    csv_files = [
        base_path / "train.csv"
    ]

    results = []

    for file in csv_files:
        if not file.exists():
            print(f"  ✗ Brak pliku: {file}")
            continue

        print(f"  ✓ Wczytuję: {file}")

        try:
            df = pd.read_csv(file)

            # Zakładam kolumnę "text" – jeśli masz inną nazwę, podaj ją!
            if "dialog" not in df.columns:
                print(f"  ✗ Plik {file} nie ma kolumny 'text'")
                continue

            for txt in df["dialog"].dropna().astype(str):
                if len(txt.strip()) > 0:
                    results.append((txt.strip(), 3))

        except Exception as e:
            print(f"  ✗ Błąd wczytywania {file}: {e}")
            continue

    print(f"  ✓ Załadowano {len(results)} rekordów z CSV")

    # Jeśli wymagany target_count, przytnie listę
    if target_count:
        return results[:target_count]

    return results


def generate_additional_label3(target_count):
    """Generuje dodatkowe przykłady Label 3 jeśli nie udało się pobrać datasetu."""
    additional_phrases = [
        "what are the best practices for",
        "can you recommend a good book about",
        "how would you approach this problem",
        "what's the difference between machine learning and deep learning",
        "explain like I'm five years old",
        "give me a step by step guide",
        "what are some common mistakes when",
        "how can I improve my skills in",
        "what resources do you suggest for learning",
        "break down this concept for me",
        "walk me through the process of",
        "what should I consider before",
        "help me understand why",
        "what are the advantages and disadvantages",
        "compare and contrast these two approaches",
        "what's your take on this topic",
        "elaborate on that point",
        "provide more context about",
        "what are some real world examples",
        "how does this apply in practice",
        "what's the history behind",
        "who are the key figures in",
        "what are the latest trends in",
        "how has this evolved over time",
        "what challenges does this face",
        "what's the future outlook for",
        "how do experts view this",
        "what are the ethical implications",
        "describe the main components of",
        "outline the key principles",
        "what metrics should I track",
        "how do I measure success in",
        "what tools are available for",
        "recommend best practices for implementing",
        "what are common pitfalls to avoid",
        "how do I get started with",
        "what prerequisites do I need",
        "break this down into simpler terms",
        "what's the big picture here",
        "connect the dots between",
        "how does this relate to",
        "what are the implications of",
        "analyze the impact of",
        "evaluate the effectiveness of",
        "critique this approach",
        "what alternatives exist",
        "how would you solve",
        "propose a solution for",
        "design a system that",
        "architect an application for",
        "model this scenario where",
    ]

    results = []
    for phrase in additional_phrases[:target_count]:
        results.append((phrase, 3))
        results.append((phrase.capitalize(), 3))
        # Dodaj wersję z pytajnikiem
        if not phrase.endswith("?"):
            results.append((phrase + "?", 3))

    return results[:target_count]


# 3. BUDOWANIE DATASETU

def build_final_dataset():
    data_greetings = []
    data_thanks = []
    data_goodbyes = []
    data_others = []

    # Generowanie wariantów dla kategorii 0, 1, 2
    for phrase in GREETINGS:
        data_greetings.extend(generate_variants(phrase, 0))

    for phrase in THANKS:
        data_thanks.extend(generate_variants(phrase, 1))

    for phrase in GOODBYES:
        data_goodbyes.extend(generate_variants(phrase, 2))

    # Generowanie kategorii 3 (Tricky + Normal)
    for phrase in TRICKY_PHRASES:
        data_others.append((phrase, 3))
        data_others.append((phrase.capitalize(), 3))

    # Obliczenie ile Label 3 potrzebujemy
    base_count = len(data_greetings) + len(data_thanks) + len(data_goodbyes)
    target_label3 = 2 * base_count
    current_label3 = len(data_others)

    print(f"\nStatystyki przed dodaniem zewnętrznych danych:")
    print(f"  Label 0: {len(data_greetings)}")
    print(f"  Label 1: {len(data_thanks)}")
    print(f"  Label 2: {len(data_goodbyes)}")
    print(f"  Label 3 (ręczne): {current_label3}")
    print(f"  Cel Label 3: {target_label3}")

    # Dodanie danych zewnętrznych do Label 3
    if current_label3 < target_label3:
        needed = target_label3 - current_label3
        external_data = load_external_data(target_count=needed)
        data_others.extend(external_data)
        print(f"  Dodano zewnętrznych: {len(external_data)}")

    final = merge_datasets(data_greetings, data_thanks, data_goodbyes, data_others)

    final.sort(key=lambda x: x[1])

    return final


# 4. ZAPIS, STATYSTYKI I WYKRES

def plot_distribution(counts):
    labels_names = ['Label 0\n(Greetings)', 'Label 1\n(Thanks)',
                    'Label 2\n(Goodbyes)', 'Label 3\n(Others)']
    values = [counts[0], counts[1], counts[2], counts[3]]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels_names, values, color=colors, edgecolor='black', linewidth=1.2)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.xlabel('Klasy', fontsize=12, fontweight='bold')
    plt.ylabel('Liczba rekordów', fontsize=12, fontweight='bold')
    plt.title('Rozkład liczby rekordów w poszczególnych klasach',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    final_dataset = build_final_dataset()
    filename = "categorized_phrases.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in final_dataset:
            writer.writerow([text, label])

    # Liczenie statystyk
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, label in final_dataset:
        counts[label] += 1

    print("\nSTATYSTYKI KOŃCOWEGO DATASETU")
    print(f"  Label 0 (Greetings):   {counts[0]} rekordów")
    print(f"  Label 1 (Thanks):      {counts[1]} rekordów")
    print(f"  Label 2 (Goodbyes):    {counts[2]} rekordów")
    print(f"  Label 3 (Others/Hard): {counts[3]} rekordów")
    print(f"\n  SUMA TOTAL: {sum(counts.values())} rekordów")

    # Rysowanie wykresu
    plot_distribution(counts)