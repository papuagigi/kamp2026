import unittest
from xray_inspection_gate import ProductInspection


class InspectionGateTest(unittest.TestCase):
    def make(self):return ProductInspection('product-1','digest-1',10.)
    def send(self,x,model,boxes,now=1.):x.accept(model,'product-1','digest-1',boxes,now)
    def test_both_models_miss_does_not_release(self):
        x=self.make();self.send(x,'first',[]);self.send(x,'second',[])
        self.assertEqual(x.state(2.,agree=True),'HOLD_NORMAL_RULE_UNVALIDATED')
    def test_second_negative_never_erases_first_warning(self):
        x=self.make();self.send(x,'first',[dict(cx=5,cy=5)]);self.send(x,'second',[])
        self.assertTrue(x.warning_preserved);self.assertEqual(x.state(2.,agree=False),'HOLD_REINSPECTION')
    def test_supervisor_deadline_does_not_wait_for_worker(self):
        x=self.make();self.send(x,'first',[])
        self.assertEqual(x.state(10.),'HOLD_DEADLINE_EXPIRED')
    def test_late_completion_does_not_cancel_hold(self):
        x=self.make();self.send(x,'first',[]);x.tick(11.);self.send(x,'second',[],12.)
        self.assertEqual(x.state(12.,agree=True),'HOLD_DEADLINE_EXPIRED')
    def test_wrong_frame_result_rejected(self):
        x=self.make()
        with self.assertRaises(ValueError):x.accept('second','product-2','digest-1',[],1.)
        self.assertEqual(x.results,{})
    def test_duplicate_does_not_replace_detection(self):
        x=self.make();self.send(x,'first',[1])
        with self.assertRaises(ValueError):self.send(x,'first',[])
        self.assertTrue(x.warning_preserved)
    def test_model_error_remains_hold(self):
        x=self.make();x.accept('first','product-1','digest-1',[],1.,'unavailable')
        self.assertEqual(x.state(2.),'HOLD_MODEL_ERROR')


if __name__=='__main__':unittest.main()
