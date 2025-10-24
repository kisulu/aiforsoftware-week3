# spacy_ner_sentiment.py
import spacy

# 1. Load a lightweight model
nlp = spacy.load("en_core_web_sm")

# 2. Create EntityRuler the modern way
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {"label": "PRODUCT", "pattern": "Kindle"},
    {"label": "PRODUCT", "pattern": "Echo Dot"},
    {"label": "BRAND", "pattern": "Amazon"},
    {"label": "BRAND", "pattern": "Sony"},
]
ruler.add_patterns(patterns)

# 3. Define word lists for simple sentiment
POSITIVE_WORDS = {"good", "great", "excellent", "love", "loved", "perfect", "best", "awesome"}
NEGATIVE_WORDS = {"bad", "terrible", "poor", "hate", "hated", "disappointed", "worst", "awful"}


def analyze_review(text):
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]

    # Simple rule-based sentiment
    tokens = [t.text.lower() for t in doc if not t.is_punct]
    score = 0
    for t in tokens:
        if t in POSITIVE_WORDS:
            score += 1
        if t in NEGATIVE_WORDS:
            score -= 1

    sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"

    return {"entities": ents, "sentiment": sentiment, "score": score}


# 4. Demo reviews
reviews = [
    "I love my new Kindle from Amazon. Battery life is excellent.",
    "The Echo Dot stopped working after a week — very disappointed.",
]

for r in reviews:
    print("\nReview:", r)
    print(analyze_review(r))