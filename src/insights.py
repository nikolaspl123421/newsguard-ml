"""Transparent diagnostics for a linear TF-IDF classification pipeline."""


def explain_prediction(pipeline, text: str, top_n: int = 8) -> dict:
    """Return model score, vocabulary coverage, and signed linear token contributions.

    Contributions explain the class score, NOT factual correctness or causality.
    """
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['model']
    vector = vectorizer.transform([text])
    score = float(pipeline.predict_proba([text])[0, 1])
    features = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    row = vector.tocsr()
    contributions = [(str(features[i]), float(v * weights[i])) for i, v in zip(row.indices, row.data)]
    positive = sorted(({'term': term, 'impact': impact} for term, impact in contributions if impact > 0), key=lambda item: item['impact'], reverse=True)[:top_n]
    negative = sorted(({'term': term, 'impact': impact} for term, impact in contributions if impact < 0), key=lambda item: item['impact'])[:top_n]
    # Share of *tokens*, not n-grams, which appear in TF-IDF's learned vocabulary.
    analyzer = vectorizer.build_tokenizer()
    preprocess = vectorizer.build_preprocessor()
    stops = vectorizer.get_stop_words() or set()
    tokens = [tok for tok in analyzer(preprocess(text)) if tok not in stops and len(tok) >= 2]
    vocabulary = vectorizer.vocabulary_
    coverage = sum(token in vocabulary for token in tokens) / len(tokens) if tokens else 0.0
    return {
        'score_true': score,
        'class_name': 'TRUE' if score >= 0.5 else 'FAKE',
        'vocabulary_coverage': float(coverage),
        'unique_features_matched': int(row.nnz),
        'top_true': positive,
        'top_fake': negative,
    }
