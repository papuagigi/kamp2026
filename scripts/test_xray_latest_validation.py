"""Invalid-coordinate handling must not turn a bad box into a good one."""
import unittest
from xray_latest_validation import partition


class GeometryGuardTests(unittest.TestCase):
    def test_preserve_order_score_and_zero_area(self):
        rows=[['a',5.,5.,2.,3.,.8],['b',6.,6.,0.,3.,.7]]
        valid,bad=partition(rows)
        self.assertEqual(valid,rows);self.assertEqual(bad,[])
    def test_quarantine_without_repair(self):
        rows=[['a',5.,5.,-2.,3.,.9],['b',6.,6.,2.,-3.,.6]]
        valid,bad=partition(rows)
        self.assertEqual(valid,[]);self.assertEqual(bad,rows)
    def test_nonfinite_is_fatal(self):
        with self.assertRaises(ValueError):partition([['a',float('nan'),5.,2.,3.,.8]])


if __name__=='__main__':unittest.main()
