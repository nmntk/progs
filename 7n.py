import nltk
nltk.download('punkt_tab')   # one-time download

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers    import Tokenizer
from sumy.summarizers.lex_rank  import LexRankSummarizer   # → Default
from sumy.summarizers.luhn      import LuhnSummarizer       # → Creative
from sumy.summarizers.lsa       import LsaSummarizer        # → Structured
from sumy.summarizers.text_rank import TextRankSummarizer   # → Diverse

text = """The Industrial Revolution took place from the 18th to 19th centuries.
It marked a shift from agrarian rural societies to industrial urban ones.
The steam engine, iron and textile industries played central roles.
It improved living standards for some but caused grim conditions for the poor."""

parser = PlaintextParser.from_string(text, Tokenizer("english"))

for name, algo in [
    ("Default", LexRankSummarizer()),
    ("Creative", LuhnSummarizer()),
    ("Structured", LsaSummarizer()),
    ("Diverse", TextRankSummarizer())
]:
    print(f"\n{name}:")
    print(" ".join(map(str, algo(parser.document, 2))))
