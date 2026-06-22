import gensim.downloader as api
from transformers import pipeline

wv  = api.load("glove-wiki-gigaword-50")
gen = pipeline("text-generation", model="gpt2")

def similar(word):
    try:    return wv.most_similar(word, topn=1)[0][0]
    except: return word

def enrich(prompt, keyword):
    return prompt.replace(keyword, similar(keyword))

def generate(prompt):
    out = gen(prompt, min_new_tokens = 100,max_new_tokens=120, do_sample=True,      
              temperature=0.85, top_p=0.9, repetition_penalty=1.2,
              return_full_text=True)[0]["generated_text"]
    text = " ".join(out.split())

    last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_end != -1:
        text = text[:last_end + 1]

    return text

keyword  = "scientist"
original = "The scientist discovered a breakthrough method to solve climate change"
enriched = enrich(original, keyword)

r1 = generate(original)
r2 = generate(enriched)

print(f"Keyword          : {keyword}  →  '{similar(keyword)}'")
print(f"Original Prompt  : {original}")
print(f"Enriched Prompt  : {enriched}")

print(f"\nOriginal Response:\n{r1}")
print(f"\nEnriched Response:\n{r2}")

print(f"Length    → Original: {len(r1)} chars | Enriched: {len(r2)} chars")
print(f"Sentences → Original: {r1.count('.')} | Enriched: {r2.count('.')}")
