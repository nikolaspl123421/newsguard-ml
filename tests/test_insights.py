import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from src.insights import explain_prediction


def test_insights_predictions_and_contributions():
    texts = [
        'official report scientific evidence confirms the discovery',
        'scientists publish verified research official journal',
        'aliens secretly rule the planet shocking conspiracy',
        'mysterious government coverup aliens conspiracy',
    ]
    target = [1, 1, 0, 0]
    pipe = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),('model', SGDClassifier(loss='log_loss', random_state=42, max_iter=2000))])
    pipe.fit(texts, target)
    result = explain_prediction(pipe, 'official science report confirms this discovery')
    assert result['class_name'] in ('TRUE','FAKE')
    assert 0 <= result['score_true'] <= 1
    assert 0 <= result['vocabulary_coverage'] <= 1
    assert result['unique_features_matched'] > 0
    assert all(x['impact'] > 0 for x in result['top_true'])
    assert all(x['impact'] < 0 for x in result['top_fake'])
    unknown = explain_prediction(pipe, 'zzzzzz qqqqqq yyyyyyy')
    assert unknown['vocabulary_coverage'] == 0
    assert unknown['unique_features_matched'] == 0
