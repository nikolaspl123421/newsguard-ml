"""Train and evaluate a text classifier on labeled news datasets."""
from pathlib import Path
import argparse
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve


def prepare_data(true_file: Path, fake_file: Path) -> pd.DataFrame:
    parts=[]
    for path, label in [(true_file,1),(fake_file,0)]:
        df=pd.read_csv(path, usecols=['title','text'])
        df['label']=label
        parts.append(df)
    df=pd.concat(parts, ignore_index=True)
    df['text_full']=(df['title'].fillna('').astype(str).str.strip()+' '+df['text'].fillna('').astype(str).str.strip()).str.strip()
    df=df[df['text_full'].str.len()>20].copy()
    # Identical texts must not appear on both sides of the train/test split.
    conflicting=df.groupby('text_full')['label'].nunique()
    conflicts=set(conflicting[conflicting>1].index)
    df=df[~df['text_full'].isin(conflicts)].drop_duplicates(subset=['text_full'])
    return df[['text_full','label']].reset_index(drop=True)


def train(df:pd.DataFrame, output:Path, max_features:int=20000):
    output.mkdir(parents=True,exist_ok=True)
    x_train,x_test,y_train,y_test=train_test_split(df['text_full'],df['label'],test_size=.2,stratify=df['label'],random_state=42)
    pipeline=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',max_features=max_features,ngram_range=(1,2),min_df=2)),('model',SGDClassifier(loss='log_loss',random_state=42,max_iter=1000,tol=1e-3))])
    pipeline.fit(x_train,y_train)
    pred=pipeline.predict(x_test)
    prob=pipeline.predict_proba(x_test)[:,1]
    precision,recall,f1,_=precision_recall_fscore_support(y_test,pred,average='binary',zero_division=0)
    metrics={'total_after_deduplication':len(df),'train_rows':len(x_train),'test_rows':len(x_test),'accuracy':float(accuracy_score(y_test,pred)),'precision_true':float(precision),'recall_true':float(recall),'f1_true':float(f1),'roc_auc':float(roc_auc_score(y_test,prob)),'confusion_matrix':confusion_matrix(y_test,pred,labels=[0,1]).tolist()}
    joblib.dump(pipeline,output/'model.joblib')
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    # Consistent dark charts for the portfolio dashboard; keep AUC precision visible.
    bg, fg, grid, teal = '#122036', '#e8f4ff', '#33475f', '#4fd1c5'
    cm=np.array(metrics['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6.3,5.2),facecolor=bg)
    ax.set_facecolor(bg)
    im=ax.imshow(cm,cmap='Blues')
    ax.set_xticks([0,1],['FAKE','TRUE'],color=fg)
    ax.set_yticks([0,1],['FAKE','TRUE'],color=fg)
    ax.set_xlabel('Predicted label',color=fg)
    ax.set_ylabel('Actual label',color=fg)
    ax.set_title('Confusion matrix',color=fg,pad=14,fontsize=14)
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,f'{v:,}',ha='center',va='center',color='white' if v>cm.max()/2 else '#11243b',weight='bold',fontsize=13)
    for spine in ax.spines.values(): spine.set_color(grid)
    fig.tight_layout();fig.savefig(output/'confusion_matrix.png',dpi=180,facecolor=bg);plt.close(fig)
    fpr,tpr,_=roc_curve(y_test,prob)
    fig,ax=plt.subplots(figsize=(6.3,5.2),facecolor=bg)
    ax.set_facecolor(bg)
    ax.plot(fpr,tpr,color=teal,lw=2.3,label=f'ROC-AUC = {metrics["roc_auc"]:.4f}')
    ax.plot([0,1],[0,1],color='#7892ac',ls='--',lw=1,label='Random baseline')
    ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    ax.set_xlabel('False positive rate',color=fg)
    ax.set_ylabel('True positive rate',color=fg)
    ax.set_title('ROC curve',color=fg,pad=14,fontsize=14)
    ax.grid(color=grid,alpha=.5)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values(): spine.set_color(grid)
    legend=ax.legend(facecolor=bg,edgecolor=grid,labelcolor=fg,loc='lower right')
    fig.tight_layout();fig.savefig(output/'roc_curve.png',dpi=180,facecolor=bg);plt.close(fig)
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--true',type=Path,default=Path('data/true.csv'))
    parser.add_argument('--fake',type=Path,default=Path('data/fake.csv'))
    parser.add_argument('--output',type=Path,default=Path('artifacts'))
    args=parser.parse_args()
    metrics=train(prepare_data(args.true,args.fake),args.output)
    print(json.dumps(metrics,indent=2))
