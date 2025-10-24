# README — ML Project: Theory & Practical Implementation

## Part 1 — Theoretical Understanding (40%)

### Q1: TensorFlow vs PyTorch
- **Philosophy:** TensorFlow uses static graphs (optimized for production), while PyTorch is dynamic and intuitive for experimentation.
- **API Style:** TensorFlow (via Keras) offers high-level APIs (`Model.fit`, `evaluate`), while PyTorch provides flexible, manual training loops.
- **Ecosystem:** TensorFlow has strong deployment tools (TF Lite, Serving, TFX). PyTorch excels in research (TorchVision, fairseq).
- **Use Case:** Choose PyTorch for experimentation and research; TensorFlow for production, mobile, and large-scale deployment.

### Q2: Jupyter Notebook Use Cases
1. **Data Exploration:** Analyze and visualize datasets interactively during preprocessing and feature engineering.
2. **Model Prototyping:** Quickly test ML models, monitor training metrics, and share reproducible experiments.

### Q3: spaCy vs Basic String Operations
- **Advanced NLP Pipeline:** spaCy provides tokenization, POS tagging, dependency parsing, and NER out-of-the-box.
- **Efficiency:** Faster and more reliable than raw Python string manipulation.
- **Rich Objects:** Access attributes (`.lemma_`, `.pos_`, `.ent_type_`) through high-level `Doc`, `Token`, and `Span` objects.

### Scikit-learn vs TensorFlow
| Aspect | Scikit-learn | TensorFlow |
|:--|:--|:--|
| **Focus** | Classical ML (SVM, Random Forest, etc.) | Deep learning (CNNs, RNNs) |
| **Ease of Use** | Simple API, great for beginners | Steeper learning curve (TF ecosystem) |
| **Community** | Mature, data-science focused | Huge, deep-learning focused |

**Summary:** Use Scikit-learn for traditional ML on structured data and TensorFlow for large-scale deep learning.

---

## Part 2 — Practical Implementation (50%)
- **Task 1:** Train a Decision Tree on the Iris dataset (handle missing values, encode labels, evaluate with accuracy, precision, recall).
- **Task 2:** Build a CNN (TensorFlow/PyTorch) for MNIST digits; target >95% accuracy; visualize 5 predictions.
- **Task 3:** Use spaCy for NER (extract product names/brands) and rule-based sentiment analysis on Amazon Reviews.

---

**Note:** For full implementation, see corresponding `.py` or Jupyter notebook files for each task with detailed code and comments.

