"""Label-free routing and conservative state decisions for the local prototype."""
import csv
import io
import pandas as pd
from xray_eval import PRED_COLUMNS, match


def canonical_predictions(stem, xyxy, width=None):
    """Use the established CSV serialization, including its NumPy precision."""
    buffer=io.StringIO(); writer=csv.writer(buffer);writer.writerow(PRED_COLUMNS)
    for x1,y1,x2,y2,score in xyxy:
        cx=(x1+x2)/2
        if width is not None:cx=width-cx
        writer.writerow([stem,cx,(y1+y2)/2,x2-x1,y2-y1,score])
    buffer.seek(0)
    return pd.read_csv(buffer)


def agreement(a,b):
    paired,_=match(a,b,iou_thr=.5,iou_only=True)
    n=int((paired.hit>=0).sum())
    return dict(matched=n,unmatched_first=len(a)-n,unmatched_second=len(b)-n,
                agree=(n==len(a)==len(b)))


def should_refer(original,flipped):
    return len(original)==0 or not agreement(original,flipped)['agree']


def outcome(first,second,referred,elapsed,deadline,error=None):
    """Never clear an alarm or return PASS. Timeout is detected after return."""
    if error:return 'REINSPECTION_ERROR'
    if elapsed>deadline:return 'REINSPECTION_TIMEOUT'
    if not referred:return 'FIRST_MODEL_DETECTION'
    if second is None:return 'REINSPECTION_ERROR'
    if not agreement(first,second)['agree']:return 'REINSPECTION_DISAGREEMENT'
    if len(first)==0:return 'REINSPECTION_NO_DETECTION'
    return 'SECOND_MODEL_AGREES_DETECTION'
