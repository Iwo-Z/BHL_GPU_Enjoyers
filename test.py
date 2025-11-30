from model import Evaluation

ev = Evaluation()
print(ev.run_optimized_LLM("Hi"))

# from textrank import TextRankSummarizer
# tr = TextRankSummarizer()
# summary = tr.summarize("Machine learning is a field of artificial intelligence that focuses on teaching. I am very simple super pretty ass gay :)", summary_percentage=0.9)
# print(summary)