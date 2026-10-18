"""NewsGuard ML v2 – educational NLP classification portfolio application."""
from pathlib import Path
import json
import html
import io
import csv
import hashlib
import numpy as np

import joblib
import pandas as pd
import streamlit as st

from src.insights import explain_prediction
from src.batch import read_news_csv, classify_rows, summarize_predictions, predictions_to_csv, CSVValidationError, MAX_BYTES, MAX_ROWS

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'artifacts'
st.set_page_config(page_title='NewsGuard ML | Portfolio Edition', page_icon='📰', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--ink:#eaf1fb;--muted:#9eacc3;--panel:#172437;--line:#2e415c;--cyan:#4fd1c5;}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at 60% -18%,#183a56 0%,#0a1524 48%,#08121f 100%);color:#eaf1fb;}
/* Preserve Streamlit sidebar controls while visually minimizing the header. */
/* Keep the Streamlit toolbar itself: it also owns the sidebar reopen button. */
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none!important;}
[data-testid="stHeader"] {height:2.75rem!important;min-height:2.75rem!important;background:transparent!important;}
/* Do not force positioning for Streamlit's responsive sidebar toggle. */
[data-testid="stSidebarCollapsedControl"] {visibility:visible!important;}
.block-container {max-width:1250px;padding-top:1.35rem!important;padding-bottom:4rem;}
h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.035em;}
h1 {font-weight:800!important;font-size:2.65rem!important;}
[data-testid="stSidebar"] {background:#0d1c2c;border-right:1px solid #23374f;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#18293e,#112036);padding:1.15rem;border:1px solid #2c415c;border-radius:16px;}
[data-testid="stMetricLabel"] {color:#a9bfd7;}
[data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;}
.stButton>button[kind="primary"] {border-radius:12px;background:#4fd1c5;color:#081521;border:none;font-weight:800;padding:.66rem 1.3rem;}
.stButton>button[kind="primary"]:hover {background:#82e8de;color:#081521;}
[data-testid="stTextArea"] textarea {background:#111f31;border:1px solid #3c536f;border-radius:14px;font-size:1rem;}
.app-masthead {display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:1rem;padding:.25rem .15rem .75rem;border-bottom:1px solid #294158;}
.app-masthead .brand {display:flex;gap:.65rem;align-items:center;font-weight:800;font-size:1rem;letter-spacing:-.015em;}
.app-masthead .brand-icon {display:inline-grid;place-items:center;width:32px;height:32px;border-radius:10px;background:#214e5d;color:#9ff1e7;}
.app-masthead .status {display:inline-block;font-size:.76rem;color:#98c9c8;letter-spacing:.055em;text-align:right;}
@media(max-width:640px){.app-masthead .status{font-size:.65rem}.hero{padding:1.3rem!important}.hero h1{font-size:2rem!important}}
.hero {border:1px solid #30516a;background:linear-gradient(120deg,#193451,#112539 60%,#173e48);padding:1.75rem 2rem;border-radius:22px;margin-bottom:1.5rem;}
.hero .eyebrow {color:#7de0d4;letter-spacing:.13em;font-size:.76rem;font-weight:800;}
.hero h1 {margin:.3rem 0 .4rem;color:#f2f7ff;}
.hero p {color:#b5ccdd;max-width:770px;font-size:1rem;margin-bottom:0;}
.result-card {border:1px solid #315a70;background:linear-gradient(110deg,#15334d,#142839);padding:1.3rem 1.6rem;border-radius:18px;margin-top:.7rem;}
.result-card .result-kicker {font-size:.78rem;letter-spacing:.1em;color:#a9cddd;font-weight:700;}
.result-card .result-label {font-size:2.3rem;font-weight:800;line-height:1.5;letter-spacing:-.03em;}
.muted {color:#9fb3c8;font-size:.9rem;}
.small-pill {display:inline-block;border:1px solid #3b6178;background:#18374a;border-radius:20px;padding:.25rem .65rem;margin:.2rem;color:#b9eced;font-size:.8rem;}
footer {visibility:hidden;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_pipeline(file: str, modified_ns: int):
    # Only load artifacts produced locally by the trusted training script.
    return joblib.load(file)

model_file = ARTIFACTS / 'model.joblib'
with st.sidebar:
    st.markdown('## 📰 NewsGuard ML')
    st.caption('NLP CLASSIFICATION LAB · v2.2')
    st.divider()
    page = st.radio('WORKSPACE', ['🔎 Analyze article', '📂 Batch CSV analysis', '📈 Model dashboard', '🧪 Challenge set', '📘 About the project'], label_visibility='visible')
    st.divider()
    st.caption('**Model:** TF-IDF + SGD logistic loss')
    st.caption('**Language:** English')
    st.caption('**Purpose:** text-pattern research, not fact-checking')

st.markdown('''<div class="app-masthead"><div class="brand"><span class="brand-icon">N</span> NewsGuard ML <span style="color:#90a6b9;font-size:.76rem;font-weight:500">/ STUDIO</span></div><span class="status">PORTFOLIO EDITION · v2.2</span></div>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">MACHINE LEARNING · NLP · EXPLAINABILITY</div>
<h1>NewsGuard ML <span style="color:#79ded4">Studio</span></h1>
<p>Explore how a text classifier labels historical news writing. Inspect predictions, model metrics, and the words influencing each decision.</p></div>''', unsafe_allow_html=True)

if not model_file.is_file():
    st.warning('Trained model not found. Open Terminal in the project folder and run `python -m src.pipeline` first.')
    st.stop()
try:
    model = load_pipeline(str(model_file), model_file.stat().st_mtime_ns)
except Exception as exc:
    st.error(f'Unable to load trained model: {exc}')
    st.stop()

if page == '🔎 Analyze article':
    st.subheader('Analyze an article')
    st.caption('Paste an English headline or article. Very short or unfamiliar text can produce misleading class scores.')
    sample = 'Scientists have confirmed that Earth orbits the Sun. Astronomers have studied planetary motion for centuries using telescopes and mathematical observations.'
    col_a, col_b = st.columns([3,1],vertical_alignment='bottom')
    with col_b:
        if st.button('Insert example',use_container_width=True):
            st.session_state['article_text'] = sample
    with col_a:
        st.caption('Demo text intentionally includes an example the classifier may get wrong.')
    text = st.text_area('ARTICLE TEXT', key='article_text', height=220, placeholder='Paste English-language article text here…')
    if st.button('Analyze text →',type='primary',disabled=not text.strip()):
        if len(text.strip()) < 30:
            st.warning('Enter at least 30 characters.')
        else:
            result = explain_prediction(model, text)
            st.session_state['last_result'] = result
            st.session_state['last_text'] = text
    if st.session_state.get('last_text') == text and 'last_result' in st.session_state:
        r = st.session_state['last_result']
        label = html.escape(r['class_name'])
        st.markdown(f'''<div class="result-card"><div class="result-kicker">PREDICTED DATASET LABEL</div>
        <div class="result-label">{'🟢' if label == 'TRUE' else '🟠'} {label}</div>
        <div class="muted">TRUE-class score: {r['score_true']:.1%} · FAKE-class score: {1-r['score_true']:.1%}</div></div>''', unsafe_allow_html=True)
        st.progress(r['score_true'],text=f'TRUE class model score · {r["score_true"]:.1%}')
        if r['vocabulary_coverage'] < .45:
            st.warning('Low vocabulary coverage: many words are unfamiliar to the model. The output needs extra caution.')
        st.caption('These scores are NOT calibrated probabilities of factual truth. A TRUE/FAKE label reflects training-data patterns only.')
        st.subheader('What influenced this prediction?')
        st.write('For this linear TF-IDF classifier, each highlighted expression contributes toward one of the learned dataset classes.')
        m1,m2 = st.columns(2)
        m1.metric('Vocabulary coverage',f'{r["vocabulary_coverage"]:.0%}',help='Fraction of token occurrences found in the model’s word vocabulary; a descriptive diagnostic, not a reliable out-of-distribution detector.')
        m2.metric('Matched model features',f'{r["unique_features_matched"]:,}')
        left,right = st.columns(2)
        with left:
            st.markdown('**Signals toward TRUE**')
            if r['top_true']:
                st.bar_chart(pd.DataFrame(r['top_true']).set_index('term')['impact'],horizontal=True,color='#4fd1c5')
            else: st.caption('No positive features matched.')
        with right:
            st.markdown('**Signals toward FAKE**')
            if r['top_fake']:
                neg = pd.DataFrame(r['top_fake'])
                neg['magnitude'] = -neg['impact']
                st.bar_chart(neg.set_index('term')['magnitude'],horizontal=True,color='#f0a66a')
            else: st.caption('No negative features matched.')
        st.caption('The feature bars show signed linear contributions. They explain the model, not whether these words make the news true or false.')

elif page == '📂 Batch CSV analysis':
    st.subheader('Batch CSV analysis')
    st.caption('Upload English news text in a UTF-8 CSV file. Maximum 10 MB and 5,000 data rows per batch. Processing runs on this app’s server.')
    uploaded = st.file_uploader('CSV file', type=['csv'], help='CSV with a header row and at least one text column. Maximum 10 MB / 5,000 data rows.')
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            frame = read_news_csv(raw)
            fingerprint = hashlib.sha256(raw).hexdigest()
            st.caption(f'{len(frame):,} rows detected · {len(frame.columns)} columns · {uploaded.size / 1024:.1f} KB')
            column = st.selectbox('Column containing news text', list(frame.columns))
            st.markdown('**Preview of uploaded data**')
            st.dataframe(frame.head(5), use_container_width=True, hide_index=True,
                         height=min(260, 54 + min(len(frame),5)*36))
            st.caption('Preview displays the first five rows. Upload data you are authorized to process.')
            current_key = (fingerprint, column)
            if st.button('Classify CSV →', type='primary'):
                with st.spinner('Classifying articles…'):
                    st.session_state['batch_results'] = classify_rows(frame, column, model)
                    st.session_state['batch_key'] = current_key
            if st.session_state.get('batch_key') == current_key and 'batch_results' in st.session_state:
                result_df = st.session_state['batch_results']
                summary = summarize_predictions(result_df)
                st.success(f'Finished processing {summary["total"]:,} rows. {summary["skipped"]:,} short or empty rows skipped.')
                a,b,c,d = st.columns(4)
                a.metric('Total rows', f'{summary["total"]:,}')
                b.metric('Predicted TRUE', f'{summary["true"]:,}')
                c.metric('Predicted FAKE', f'{summary["fake"]:,}')
                d.metric('Skipped', f'{summary["skipped"]:,}')
                st.markdown('**Results preview**')
                st.dataframe(result_df.head(100), use_container_width=True, hide_index=True)
                if len(result_df) > 100:
                    st.caption('Showing the first 100 rows; download the CSV to view all results.')
                st.download_button('Download complete results (CSV)',
                                   data=predictions_to_csv(result_df),
                                   file_name='newsguard_predictions.csv', mime='text/csv')
                st.caption('Export preserves all original CSV columns and adds row number, dataset prediction and TRUE-class score. The score is not a factual truth probability.')
            elif st.session_state.get('batch_key') is not None:
                st.info('The file or selected text column changed. Click Classify CSV to generate updated predictions.')
        except CSVValidationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f'Batch prediction failed ({type(exc).__name__}). Check that the model has been trained and the text column contains valid strings.')

elif page == '📈 Model dashboard':
    st.subheader('Model performance')
    metrics_file = ARTIFACTS / 'metrics.json'
    if not metrics_file.exists():
        st.warning('No evaluation metrics. Run `python -m src.pipeline`.')
    else:
        metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
        a,b,c,d = st.columns(4)
        a.metric('Accuracy',f'{metrics["accuracy"]:.2%}')
        b.metric('Precision · TRUE',f'{metrics["precision_true"]:.2%}')
        c.metric('Recall · TRUE',f'{metrics["recall_true"]:.2%}')
        d.metric('F1 · TRUE',f'{metrics["f1_true"]:.2%}')
        st.markdown('')
        a,b,c = st.columns(3)
        a.metric('Deduplicated records',f'{metrics["total_after_deduplication"]:,}')
        b.metric('Train records',f'{metrics["train_rows"]:,}')
        c.metric('Test records',f'{metrics["test_rows"]:,}')
        st.caption(f'ROC–AUC: {metrics["roc_auc"]:.4f} · stratified random 80/20 split · random seed 42')
        cm = metrics['confusion_matrix']
        st.subheader('Confusion matrix')
        grid = pd.DataFrame(cm,index=['Actual FAKE','Actual TRUE'],columns=['Predicted FAKE','Predicted TRUE'])
        st.dataframe(grid,use_container_width=True)
        st.caption('Rows are actual dataset labels; columns are predicted classes.')
        left,right = st.columns(2)
        with left:
            if (ARTIFACTS/'confusion_matrix.png').exists():
                st.image(str(ARTIFACTS/'confusion_matrix.png'),caption='Confusion matrix from held-out dataset')
        with right:
            if (ARTIFACTS/'roc_curve.png').exists():
                st.image(str(ARTIFACTS/'roc_curve.png'),caption='Receiver operating characteristic')
        st.info('This evaluation is within the same source dataset. Strong scores do not establish reliable performance on new publishers, new topics, or fact verification.')

elif page == '🧪 Challenge set':
    st.subheader('Out-of-sample behavior lab')
    st.caption('Handwritten examples not intended as a representative external benchmark. Use this page to demonstrate failure cases and dataset bias.')
    examples = [
        {'text':'Scientists have confirmed that Earth orbits the Sun. Astronomers have studied planetary motion for centuries using telescopes and mathematical observations.', 'note':'Established astronomical fact; previously misclassified as FAKE.'},
        {'text':'The Pacific Ocean is larger than the Atlantic Ocean and spans parts of both the Northern and Southern Hemispheres.', 'note':'Established geographic fact.'},
        {'text':'According to a fictional press release, invisible dragons secretly control every weather satellite on Earth and send coded messages through clouds.', 'note':'Intentionally fictional statement.'},
        {'text':'A local council announced that road repairs will begin next Monday, according to a short community notice.', 'note':'Unverified illustrative news-style claim.'},
    ]
    results=[]
    for item in examples:
        r=explain_prediction(model,item['text'])
        results.append({'Illustrative text':item['text'],'Context (not evaluation label)':item['note'],'Dataset prediction':r['class_name'],'TRUE score':round(r['score_true'],3),'Vocabulary coverage':round(r['vocabulary_coverage'],3)})
    st.dataframe(pd.DataFrame(results),hide_index=True,use_container_width=True,column_config={
        'Illustrative text':st.column_config.TextColumn('Illustrative text',width='large',help='Select a cell to read the complete text'),
        'Context (not evaluation label)':st.column_config.TextColumn('Context (not evaluation label)',width='large'),
        'Dataset prediction':st.column_config.TextColumn('Prediction',width='small'),
        'TRUE score':st.column_config.NumberColumn('TRUE score',format='%.3f'),
        'Vocabulary coverage':st.column_config.NumberColumn('Coverage',format='%.3f')
    })
    with st.expander('Read full challenge texts'):
        for i,item in enumerate(results,1):
            st.markdown(f'**Example {i} — {item["Dataset prediction"]}**')
            st.write(item['Illustrative text'])
            st.caption(item['Context (not evaluation label)'])
    st.download_button('Export example predictions (CSV)',data=pd.DataFrame(results).to_csv(index=False).encode('utf-8'),file_name='newsguard_challenge_examples.csv',mime='text/csv')
    st.warning('These four constructed texts are a demonstration, NOT a valid accuracy test. To measure generalization, use independently sourced, verified and properly labeled articles.')

else:
    st.subheader('About this project')
    st.markdown('''**NewsGuard ML Studio 2.2** is a portfolio-ready educational NLP application built around an undergraduate fake-news classification project.

**Pipeline:** article headline + body → TF-IDF (unigrams/bigrams) → linear classifier (`SGDClassifier`, logistic loss) → dataset-class output.

**Interpretability:** signed token contributions explain which terms influenced the linear model. Vocabulary coverage is only a rough descriptive diagnostic.

**Evaluation:** duplicate and conflicting text labels are removed *before* train/test splitting. The reported metrics come from a random stratified holdout within the original dataset; they do not evaluate transfer to other publishers or time periods.

**Limitations:** the system cannot consult live sources, verify factual claims, or determine truth. Outputs should not be used for consequential moderation or fact-checking decisions.

**Batch analysis:** UTF-8 CSV upload (10 MB), selectable text column, preview, up to 5,000 records, summary metrics, and export retaining original source columns. No external factual sources are consulted.

**Stack:** Python · pandas · scikit-learn · Streamlit · matplotlib · pytest.''')
    st.caption('Source of the classification dataset: original user-provided true.csv and fake.csv. Confirm redistribution rights before publishing these datasets publicly.')
