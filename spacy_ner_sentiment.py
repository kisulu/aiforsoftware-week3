# spacy_ner_sentiment.py
import spacy
from spacy.pipeline import EntityRuler


# 1. Load a lightweight model (change to 'en_core_web_trf' if available and you want better accuracy)
nlp = spacy.load('en_core_web_sm')


# 2. Add rule-based patterns to capture product names/brands (EntityRuler)
ruler = EntityRuler(nlp, overwrite_ents=True)
patterns = [
{"label": "PRODUCT", "pattern": "Kindle"},
{"label": "PRODUCT", "pattern": "Echo Dot"},
{"label": "BRAND", "pattern": "Amazon"},
{"label": "BRAND", "pattern": "Sony"},
]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler, before='ner')


# 3. Example function to extract entities and do rule-based sentiment
POSITIVE_WORDS = {"good","great","excellent","love","loved","perfect","best","awesome"}
NEGATIVE_WORDS = {"bad","terrible","poor","hate","hated","disappointed","worst","awful"}




def analyze_review(text):
doc = nlp(text)
ents = [(ent.text, ent.label_) for ent in doc.ents]


# Simple rule-based sentiment: token matching
tokens = [t.text.lower() for t in doc if not t.is_punct]
score = 0
for t in tokens:
if t in POSITIVE_WORDS:
score += 1
if t in NEGATIVE_WORDS:
score -= 1
sentiment = 'positive' if score>0 else 'negative' if score<0 else 'neutral'


return {'entities': ents, 'sentiment': sentiment, 'score': score}


# 4. Demo
reviews = [
"I love my new Kindle from Amazon. Battery life is excellent.",
"The Echo Dot stopped working after a week — very disappointed.",
]
for r in reviews:
print(r)
print(analyze_review(r))


# Notes: Off-the-shelf NER models may not reliably extract product names/brands. Use EntityRuler, or fine-tune a model on labeled data for better recall/precision.