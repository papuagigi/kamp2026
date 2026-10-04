import unittest
import numpy as np
import pandas as pd
from xray_eval import PRED_COLUMNS
from xray_cascade_runtime import canonical_predictions,should_refer,outcome


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.a=pd.DataFrame([['s',10,10,4,4,.9]],columns=PRED_COLUMNS)
        self.empty=self.a.iloc[:0]
    def test_csv_threshold_precision(self):
        p=canonical_predictions('s',np.array([[0,0,4,4,.68106407]],dtype=np.float32))
        self.assertEqual(p.score.iloc[0],.68106407)
    def test_flip_coordinates(self):
        p=canonical_predictions('s',np.array([[2,3,6,9,.8]],dtype=np.float32),20)
        self.assertEqual(p.cx.iloc[0],16);self.assertEqual(p.cy.iloc[0],6)
    def test_empty_must_refer(self):self.assertTrue(should_refer(self.empty,self.empty))
    def test_stable_boxes(self):self.assertFalse(should_refer(self.a,self.a))
    def test_no_pass_on_empty_agreement(self):
        self.assertEqual(outcome(self.empty,self.empty,True,.1,1),'REINSPECTION_NO_DETECTION')
    def test_second_cannot_clear_first(self):
        self.assertEqual(outcome(self.a,self.empty,True,.1,1),'REINSPECTION_DISAGREEMENT')
        self.assertEqual(len(self.a),1)
    def test_timeout_with_good_predictions(self):
        self.assertEqual(outcome(self.a,self.a,True,2,1),'REINSPECTION_TIMEOUT')
    def test_exception_with_good_first(self):
        self.assertEqual(outcome(self.a,None,True,.1,1,'model failed'),'REINSPECTION_ERROR')
    def test_missing_second(self):
        self.assertEqual(outcome(self.a,None,True,.1,1),'REINSPECTION_ERROR')


if __name__=='__main__':unittest.main()
