# pip install gensim transformers nltk
import gensim.downloader as api
from transformers import pipeline
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Load models
wv = api.load("glove-wiki-gigaword-100")
gen = pipeline("text-generation", model="gpt2")

# Function to enrich prompt
def enrich(prompt, keyword):
    words = word_tokenize(prompt)
    new_words = []
    
    for w in words:
        if w.lower() == keyword.lower():
            try:
                new_words.append(wv.most_similar(w.lower())[0][0])
            except:
                new_words.append(w)
        else:
            new_words.append(w)
    
    return " ".join(new_words)

# Function to generate text
def generate(p):
    return gen(p, max_new_tokens=50, pad_token_id=50256)[0]['generated_text']

# Input
prompt = "The scientist discovered a method to solve climate change"
keyword = "scientist"

# Process
enriched = enrich(prompt, keyword)

# Generate outputs
r1 = generate(prompt)
r2 = generate(enriched)

# Results
print("Original Prompt:", prompt)
print("Enriched Prompt:", enriched)

print("\nOriginal Output:\n", r1)
print("\nEnriched Output:\n", r2)

print("\nLength Comparison:", len(r1), "vs", len(r2))
print("Sentence Count:", r1.count("."), "vs", r2.count(".")
