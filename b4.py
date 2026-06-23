# pip install gensim transformers
# pip install transformers==4.57.6

import gensim.downloader as api
from transformers import pipeline

# Load word embeddings and GPT-2
wv = api.load("glove-wiki-gigaword-50")
gen = pipeline("text-generation", model="gpt2")

# Original prompt
prompt = "Who is king?"

# Get similar word and enrich prompt
similar_word = wv.most_similar("king", topn=1)[0][0]
new_prompt = prompt.replace("king", similar_word)

print("Original Prompt :", prompt)
print("Enriched Prompt :", new_prompt)

# Generate responses
out1 = gen(prompt, max_length=50)[0]["generated_text"]
out2 = gen(new_prompt, max_length=50)[0]["generated_text"]

print("\nOriginal Output:\n", out1)
print("\nEnriched Output:\n", out2)

# Compare outputs
print("\nOriginal Length :", len(out1))
print("Enriched Length :", len(out2))