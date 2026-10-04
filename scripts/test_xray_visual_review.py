"""Check visual-decision completeness, duplicates, and unresolved-score bounds."""
import copy
import unittest

from xray_eval import visual_review_counts


class VisualReviewTests(unittest.TestCase):
    def data(self, tokens=('p1',), matches=('p1',), uncertain=()):
        cases=[dict(case='T001',split='test',stem='a',gt_index=0,figure_sha256='hash',
            panels={'A':dict(run='model',predictions=[dict(token=t,pred_index=i) for i,t in enumerate(tokens)])})]
        decisions=[dict(case='T001',panel='A',target_visible='yes',matches=list(matches),
            nonmatches=[t for t in tokens if t not in matches and t not in uncertain],
            uncertain=list(uncertain),evidence_sha256='hash')]
        return cases,decisions

    def test_one_confirmed(self):
        r=visual_review_counts(*self.data())[0]
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['f1']),(1,0,0,1.))
        self.assertIsNone(r['ap'])

    def test_duplicate_is_extra_fp(self):
        r=visual_review_counts(*self.data(('p1','p2'),('p1','p2')))[0]
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['duplicate_predictions']),(1,1,0,1))

    def test_no_box_is_miss(self):
        r=visual_review_counts(*self.data((),()))[0]
        self.assertEqual((r['tp'],r['fp'],r['fn']),(0,0,1))
        self.assertEqual(r['images_with_no_selected_box'],1)

    def test_wrong_box_is_fp_and_fn(self):
        r=visual_review_counts(*self.data(matches=()))[0]
        self.assertEqual((r['tp'],r['fp'],r['fn']),(0,1,1))

    def test_uncertain_not_silently_positive(self):
        r=visual_review_counts(*self.data(matches=(),uncertain=('p1',)))[0]
        self.assertIsNone(r['f1'])
        self.assertEqual((r['lower_bound']['f1'],r['upper_bound']['f1']),(0.,1.))

    def test_missing_or_duplicate_decision(self):
        c,d=self.data()
        for rows in [[],d+d]:
            with self.assertRaises(ValueError):visual_review_counts(c,rows)

    def test_missing_or_duplicate_box_decision(self):
        c,d=self.data()
        for matches in [[],['p1','p1']]:
            bad=copy.deepcopy(d);bad[0]['matches']=matches
            with self.assertRaises(ValueError):visual_review_counts(c,bad)

    def test_evidence_change_rejected(self):
        c,d=self.data();d[0]['evidence_sha256']='different'
        with self.assertRaises(ValueError):visual_review_counts(c,d)

    def test_ambiguous_reference_cannot_be_tp(self):
        c,d=self.data();d[0]['target_visible']='uncertain'
        with self.assertRaises(ValueError):visual_review_counts(c,d)

    def test_two_targets_do_not_share_one_prediction(self):
        c,d=self.data();c2=copy.deepcopy(c[0]);c2['case']='T002';c2['gt_index']=1
        d2=copy.deepcopy(d[0]);d2['case']='T002'
        with self.assertRaises(ValueError):visual_review_counts(c+[c2],d+[d2])

    def test_duplicate_case_identifier(self):
        c,d=self.data();c2=copy.deepcopy(c[0]);c2['gt_index']=1
        with self.assertRaises(ValueError):visual_review_counts(c+[c2],d)


if __name__=='__main__':unittest.main()
