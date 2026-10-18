"""CSV import, predictable batch inference and user-friendly exported results."""
from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd

MAX_ROWS = 5_000
MAX_BYTES = 10 * 1024 * 1024
MIN_TEXT_LENGTH = 30

class CSVValidationError(ValueError):
    """An understandable input validation failure."""


def read_news_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise CSVValidationError('The CSV file is empty.')
    if len(content) > MAX_BYTES:
        raise CSVValidationError('File exceeds the 10 MB upload limit.')
    try:
        # UTF-8 BOM is common in Excel exports. Keep all fields as text.
        frame = pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False,
                            encoding='utf-8-sig', nrows=MAX_ROWS + 1)
    except UnicodeDecodeError as exc:
        raise CSVValidationError('CSV must use UTF-8 encoding. Export it as CSV UTF-8 and try again.') from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise CSVValidationError('Unable to parse CSV. Check the header, separator and quoting.') from exc
    if frame.empty:
        raise CSVValidationError('No data rows found. Add a header and at least one row.')
    if len(frame) > MAX_ROWS:
        raise CSVValidationError(f'More than {MAX_ROWS:,} rows. Split the file into smaller batches.')
    if any(str(c).startswith('Unnamed:') for c in frame.columns):
        raise CSVValidationError('CSV contains an unnamed column. Give each column a header.')
    if len(set(frame.columns)) != len(frame.columns):
        raise CSVValidationError('Column names must be unique.')
    return frame


def classify_rows(frame: pd.DataFrame, text_column: str, model) -> pd.DataFrame:
    if text_column not in frame.columns:
        raise CSVValidationError('Selected text column was not found in CSV.')
    series = frame[text_column].fillna('').astype(str)
    eligible = series.str.strip().str.len().ge(MIN_TEXT_LENGTH)
    scores = np.full(len(frame), np.nan, dtype=float)
    if eligible.any():
        probabilities = np.asarray(model.predict_proba(series.loc[eligible].tolist()))
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError('Model is missing the TRUE (1) class.')
        scores[eligible.to_numpy()] = probabilities[:, classes.index(1)]
    labels = np.where(np.isnan(scores), 'SKIPPED (<30 characters)',
                      np.where(scores >= 0.5, 'TRUE', 'FAKE'))
    # Preserve the original text and remaining uploaded columns, without overwriting
    # a source column that happens to have the same name as an output column.
    exported = frame.copy().reset_index(drop=True)
    for key in ('row_number','predicted_dataset_class','true_class_score'):
        if key in exported.columns:
            exported = exported.rename(columns={key:f'original_{key}'})
    exported.insert(0, 'row_number', np.arange(1, len(frame)+1))
    exported['predicted_dataset_class'] = labels
    exported['true_class_score'] = np.round(scores, 4)
    return exported


def summarize_predictions(result: pd.DataFrame) -> dict[str, int]:
    counts = result['predicted_dataset_class'].value_counts()
    return {'total':len(result), 'true':int(counts.get('TRUE',0)),
            'fake':int(counts.get('FAKE',0)),
            'skipped':int(result['predicted_dataset_class'].str.startswith('SKIPPED').sum())}


def predictions_to_csv(result: pd.DataFrame) -> bytes:
    return result.to_csv(index=False, na_rep='', lineterminator='\n').encode('utf-8-sig')
