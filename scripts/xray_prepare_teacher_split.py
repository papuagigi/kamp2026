"""Reserve groups inside the official TRAIN split to calibrate pseudo-label acceptance."""
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from xray_config import ROOT,DATA
from xray_prepare_roadmap import OUT,link


def main():
    directory=OUT/'variants/teacher';
    if directory.exists():raise RuntimeError('Teacher split already exists')
    man=pd.read_csv(DATA/'manifest.csv').fillna('');train=man[man.split=='train'].copy()
    bad=set(train.loc[train.label_issue.str.contains('라벨누락'),'stem'])
    train=train[~train.stem.isin(bad)].reset_index(drop=True)
    groups=train.burst_id;strata=train.machine+'_'+train.card_type
    fitter,cal=next(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(train,strata,groups))
    fit=train.iloc[fitter];hold=train.iloc[cal]
    assert set(fit.burst_id).isdisjoint(hold.burst_id)
    assert set(hold.burst_id).isdisjoint(man.loc[man.split!='train','burst_id'])
    for split,part in [('train',fit),('val',hold)]:
        for r in part.itertuples():
            link(OUT/'images/clean'/f'{r.stem}.png',directory/'images'/split/f'{r.stem}.png')
            link(DATA/'labels/train'/f'{r.stem}.txt',directory/'labels'/split/f'{r.stem}.txt')
    (directory/'data.yaml').write_text(yaml.safe_dump({'path':'.',
          'train':'images/train','val':'images/val','names':{0:'defect'}}))
    report=ROOT/'reports/roadmap_20261002';report.mkdir(parents=True,exist_ok=True)
    (report/'teacher_calibration_stems.txt').write_text('\n'.join(hold.stem)+'\n')
    result={'seed':0,'fit_images':len(fit),'calibration_images':len(hold),'fit_groups':fit.burst_id.nunique(),
            'calibration_groups':hold.burst_id.nunique(),'excluded_missing_label':sorted(bad),
            'fit_stems':fit.stem.tolist(),'calibration_stems':hold.stem.tolist(),
            'source':'official train split only; external validation/test excluded'}
    (report/'teacher_split.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('stems')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
