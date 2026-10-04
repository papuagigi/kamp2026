"""Offline product-state prototype. Never controls a PLC or releases a product.

The supervisor is independent of detector work. Time is supplied by the caller.
Each product and frame digest must match before a result can be attached.
"""
from dataclasses import dataclass,field


@dataclass
class ProductInspection:
    product_id:str
    frame_sha256:str
    deadline:float
    results:dict=field(default_factory=dict)
    events:list=field(default_factory=list)
    hold_reason:str|None=None

    def accept(self,model,product_id,frame_sha256,boxes,now,error=None):
        if product_id!=self.product_id or frame_sha256!=self.frame_sha256:
            raise ValueError('Product or frame mismatch; result rejected')
        if model not in ('first','second'):raise ValueError('Unknown detector stage')
        if model in self.results:raise ValueError('Duplicate result; original preserved')
        self.tick(now)
        self.results[model]=dict(boxes=boxes,error=error,received=now)
        self.events.append(dict(model=model,received=now,boxes=boxes,error=error))
        if error:self.hold_reason=self.hold_reason or 'MODEL_ERROR'

    def tick(self,now):
        if now>=self.deadline and len(self.results)<2:
            self.hold_reason=self.hold_reason or 'DEADLINE_EXPIRED'

    def state(self,now,agree=None):
        self.tick(now)
        if self.hold_reason:return 'HOLD_'+self.hold_reason
        if len(self.results)<2:return 'WAIT_SECOND_RESULT'
        if agree is not True:return 'HOLD_REINSPECTION'
        if any(r['boxes'] for r in self.results.values()):return 'HOLD_DETECTED_OBJECT'
        return 'HOLD_NORMAL_RULE_UNVALIDATED'

    @property
    def warning_preserved(self):
        return any(bool(r['boxes']) for r in self.results.values())
