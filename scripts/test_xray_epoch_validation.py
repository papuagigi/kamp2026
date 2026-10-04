"""Selection-policy tests: protect model choice from metric/tie mistakes."""
import unittest
from xray_epoch_validation import better

def record(epoch,f1,recall,ap75):
    return dict(epoch=epoch,metrics=dict(iou50=dict(f1=f1,recall=recall),iou75=dict(ap=ap75)))

class EpochSelectionTests(unittest.TestCase):
    def test_primary_f1_wins(self):
        self.assertTrue(better(record(2,.9,.8,.2),record(1,.8,1.,.9)))
    def test_recall_breaks_f1_tie(self):
        self.assertTrue(better(record(2,.9,.95,.2),record(1,.9,.9,.9)))
    def test_ap75_breaks_recall_tie(self):
        self.assertTrue(better(record(2,.9,.95,.4),record(1,.9,.95,.3)))
    def test_tolerance_prefers_earlier(self):
        self.assertFalse(better(record(2,.9+1e-10,.95,.3),record(1,.9,.95,.3)))
    def test_empty_prediction_score_does_not_win(self):
        self.assertFalse(better(record(2,0,0,0),record(1,.1,.1,.1)))

if __name__=='__main__':unittest.main()
