import pandas as pd
from src.pipeline import prepare_data

def test_deduplication_and_conflicting_labels(tmp_path):
    true=pd.DataFrame({'title':['Real A','Real A','Conflict'],'text':['A sufficiently long example text','A sufficiently long example text','Shared article with enough words']})
    fake=pd.DataFrame({'title':['Fake B','Conflict'],'text':['Another sufficiently long example','Shared article with enough words']})
    true.to_csv(tmp_path/'true.csv',index=False)
    fake.to_csv(tmp_path/'fake.csv',index=False)
    data=prepare_data(tmp_path/'true.csv',tmp_path/'fake.csv')
    assert len(data)==2
    assert set(data.label)=={0,1}
