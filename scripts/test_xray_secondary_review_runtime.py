import unittest
import pandas as pd
from xray_eval import PRED_COLUMNS
from xray_secondary_review_runtime import review_required,review_outcome


def boxes(values):
    return pd.DataFrame([['example',*v] for v in values],columns=PRED_COLUMNS)


class ReviewTests(unittest.TestCase):
    def test_same_wrong_nonempty_prediction_always_reviewed_in_all_scope(self):
        a=boxes([[15.,15.,10.,10.,.99]])
        self.assertFalse(review_required('flip',a,a))
        self.assertTrue(review_required('all',a))
    def test_no_detection_is_not_pass(self):
        a=boxes([])
        self.assertTrue(review_required('flip',a,a))
        self.assertEqual(review_outcome(a,a,True,2.),'REINSPECTION_NO_DETECTION')
    def test_second_model_does_not_clear_first_warning(self):
        a=boxes([[15.,15.,10.,10.,.99]])
        self.assertEqual(review_outcome(a,boxes([]),True,2.),'REINSPECTION_DISAGREEMENT')
    def test_agreement_has_no_truth_or_pass_claim(self):
        a=boxes([[15.,15.,10.,10.,.99]])
        self.assertEqual(review_outcome(a,a,True,2.),'TWO_MODELS_CORRESPOND_NOT_TRUTH_CONFIRMED')
    def test_missing_flip_is_invalid_configuration(self):
        with self.assertRaises(ValueError):review_required('flip',boxes([]))
    def test_optional_budget_never_turns_error_into_pass(self):
        a=boxes([])
        self.assertEqual(review_outcome(a,a,True,2.,deadline=1.),'REINSPECTION_TIMEOUT')
        self.assertEqual(review_outcome(a,None,True,.1,error='invalid geometry'),'REINSPECTION_ERROR')


if __name__=='__main__':unittest.main()
