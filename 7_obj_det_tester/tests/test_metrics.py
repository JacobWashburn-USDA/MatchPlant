import importlib.util
from pathlib import Path
import sys
import unittest
import torch

path=Path(__file__).resolve().parents[1]/'test.py'
spec=importlib.util.spec_from_file_location('matchplant_detection_test',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MetricsChecks(unittest.TestCase):
    def test_duplicate_predictions_cannot_match_one_object_twice(self):
        boxes=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]])
        result=module.analyze_size_performance(boxes,torch.tensor([.9,.8]),boxes[:1])
        self.assertEqual(result['small']['tp'],1)
        self.assertEqual(result['small']['fp'],1)
        self.assertEqual(result['small']['fn'],0)

    def test_degenerate_box_has_zero_iou(self):
        box=torch.tensor([0.,0.,0.,0.])
        self.assertEqual(module.calculate_iou(box,box),0)


if __name__ == '__main__':
    unittest.main()
