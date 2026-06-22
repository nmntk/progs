#pip install transformers==4.57.6 torch
from transformers import pipeline, set_seed

set_seed(42)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Machine learning (ML) is a branch of artificial intelligence (AI) focused on building
systems that learn from data and improve their performance over time without being
explicitly programmed. Traditional programming requires developers to write specific
rules for every task. In contrast, machine learning algorithms identify patterns in
data and use those patterns to make predictions or decisions. 

There are three main types of machine learning: supervised learning, where the model is trained on labeled
data; unsupervised learning, where the model finds hidden patterns in unlabeled data;
and reinforcement learning, where an agent learns by interacting with an environment
and receiving rewards or penalties. 

Machine learning is used in many real-worldapplications such as email spam filtering, recommendation systems, image recognition,
natural language processing, and medical diagnosis. Deep learning, a subset of machine
learning, uses neural networks with many layers to solve complex problems like speech
recognition and autonomous driving. As the volume of data grows and computing power
increases, machine learning is becoming one of the most transformative technologies
of our time.
"""

print("Original:\n", text)

s1 = summarizer(text, max_length=50, min_length=20,
                do_sample=False)

s2 = summarizer(text, max_length=50, min_length=20,
                do_sample=True,
                temperature=2.0)

s3 = summarizer(text, max_length=50, min_length=20,
                do_sample=True,
                temperature=2.0,
                top_k=100)

s4 = summarizer(text, max_length=50, min_length=20,
                do_sample=True,
                temperature=2.0,
                top_k=100,
                top_p=0.95)

print("\nDefault :\n", s1[0]["summary_text"])
print("\nCreative :\n", s2[0]["summary_text"])
print("\nStructured 3:\n", s3[0]["summary_text"])
print("\nDiverse 4:\n", s4[0]["summary_text"])
