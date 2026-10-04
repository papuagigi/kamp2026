"""Accuracy-first review state logic. Never equate agreement with correctness."""
from xray_cascade_runtime import should_refer,outcome


def review_required(scope,original,flipped=None):
    if scope=='all':return True
    if scope!='flip':raise ValueError('Unknown review scope')
    if flipped is None:raise ValueError('Flip predictions are required')
    return should_refer(original,flipped)


def review_outcome(first,second,referred,elapsed,deadline=None,error=None):
    state=outcome(first,second,referred,elapsed,float('inf') if deadline is None else deadline,error)
    if state=='SECOND_MODEL_AGREES_DETECTION':
        return 'TWO_MODELS_CORRESPOND_NOT_TRUTH_CONFIRMED'
    return state
