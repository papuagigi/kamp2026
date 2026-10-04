"""Referral rules must handle silence and confident mistakes without using labels."""
import unittest
import pandas as pd
from xray_cascade_audit import referral_sets


class ReferralTest(unittest.TestCase):
    def test_missing_prediction_and_threshold_boundary(self):
        raw=pd.DataFrame({'stem':['low','boundary','confident'], 'score':[.2,.7,.95]})
        gates=referral_sets(raw, ['silent','low','boundary','confident'], .7, .05, 0, .2)
        self.assertEqual(gates['no_detection'], {'silent','low'})
        self.assertEqual(gates['score_margin'], {'silent','low','boundary'})
        self.assertNotIn('confident',gates['score_margin'])

    def test_confident_box_can_hide_another_missed_target(self):
        raw=pd.DataFrame({'stem':['partial'], 'score':[.95]})
        gates=referral_sets(raw, ['partial'], .7, .05, 0, 0)
        self.assertEqual(gates['score_margin'],set())
        self.assertEqual(gates['all_images_control'],{'partial'})

    def test_audit_is_reproducible_and_does_not_read_ground_truth(self):
        raw=pd.DataFrame({'stem':[str(i) for i in range(20)], 'score':[.95]*20})
        a=referral_sets(raw,list(raw.stem),.7,.05,0,.2)
        b=referral_sets(raw,list(reversed(raw.stem)),.7,.05,0,.2)
        self.assertEqual(a,b)
        self.assertEqual(len(a['score_margin_plus_sample']),4)


if __name__=='__main__':unittest.main()
