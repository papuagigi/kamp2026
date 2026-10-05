import unittest
import numpy as np
from xray_review_geometry import around, tiles, clipped_labels, remap, suppress


class ReviewGeometryTests(unittest.TestCase):
    def test_tiles_cover_entire_image(self):
        for w,h in [(64,80),(316,332),(576,444),(512,512)]:
            mask=np.zeros((h,w),dtype=bool)
            for x0,y0,x1,y1 in tiles(w,h):mask[y0:y1,x0:x1]=True
            self.assertTrue(mask.all())
    def test_crop_near_edge_keeps_dimensions(self):
        self.assertEqual(around(3,3,316,332),(0,0,96,96))
        self.assertEqual(around(315,331,316,332),(220,236,316,332))
    def test_partial_labels_cannot_become_empty(self):
        labels,partial=clipped_labels([[95,50,10,10]],(0,0,96,96))
        self.assertTrue(partial);self.assertEqual(len(labels),1)
    def test_coordinates_return_to_parent(self):
        x=remap([[10,20,30,40,.8]],(100,200,196,296))
        np.testing.assert_allclose(x,[[110,220,130,240,.8]])
    def test_duplicate_tile_predictions_suppressed(self):
        self.assertEqual(len(suppress([[10,10,20,20,.8],[10,10,20,20,.9]])),1)

if __name__=='__main__':unittest.main()
