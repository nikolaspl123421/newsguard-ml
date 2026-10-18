from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from src.batch import (CSVValidationError, read_news_csv, classify_rows,
                       summarize_predictions, predictions_to_csv)


class DemoModel:
    classes_ = np.array([0,1])
    def predict_proba(self, values):
        return np.array([[0.2,0.8] if 'positive' in x else [0.9,0.1] for x in values])


def test_import_and_export_preserves_text_and_extra_columns():
    content = ('text,source\n"A sufficiently long positive test news article.",publisher A\n'
               '"A sufficiently long negative test news article.",publisher B\n'
               '"",publisher C\n').encode()
    frame=read_news_csv(content)
    result=classify_rows(frame,'text',DemoModel())
    assert result['predicted_dataset_class'].tolist()==['TRUE','FAKE','SKIPPED (<30 characters)']
    assert summarize_predictions(result)=={'total':3,'true':1,'fake':1,'skipped':1}
    recovered=pd.read_csv(BytesIO(predictions_to_csv(result)),keep_default_na=False)
    assert recovered['text'].iloc[0]==frame['text'].iloc[0]
    assert recovered['source'].tolist()==['publisher A','publisher B','publisher C']
    assert recovered['row_number'].tolist()==[1,2,3]
    assert recovered['true_class_score'].iloc[2]==''


def test_empty_csv_and_large_file_rejected():
    with pytest.raises(CSVValidationError,match='empty'):
        read_news_csv(b'')
    with pytest.raises(CSVValidationError,match='10 MB'):
        read_news_csv(b'x'*(10*1024*1024+1))


def test_bad_encoding_and_more_than_5000_rows_rejected():
    with pytest.raises(CSVValidationError,match='UTF-8'):
        read_news_csv(b'text\n\xff')
    contents=('text\n'+('hello world example text number 100\n'*5001)).encode()
    with pytest.raises(CSVValidationError,match='5,000'):
        read_news_csv(contents)


def test_preserves_original_conflicting_column_names():
    frame=pd.DataFrame({'text':['A sufficiently long positive news item'],
                        'row_number':['original row'],
                        'true_class_score':['source rating']})
    result=classify_rows(frame,'text',DemoModel())
    assert result.loc[0,'original_row_number']=='original row'
    assert result.loc[0,'original_true_class_score']=='source rating'
    assert result.loc[0,'row_number']==1
