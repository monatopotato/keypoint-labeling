import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pose_inference import box_iou, merge_candidates, result_candidates, tile_windows, predict_person_zoom


class DistantPoseTests(unittest.TestCase):
    def test_person_zoom_keeps_tile_only_subject_after_full_frame_detection(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Optional YOLO image dependencies not installed')
        picture = BytesIO()
        Image.new('RGB', (640, 400)).save(picture, format='PNG')
        picture.seek(0)

        def detected(box):
            result = Mock()
            result.boxes.xyxy.tolist.return_value = [box]
            result.boxes.conf.tolist.return_value = [.9]
            return [result]

        def posed(box, joint):
            result = Mock()
            result.boxes.xyxy.cpu.return_value.tolist.return_value = [box]
            result.boxes.conf.cpu.return_value.tolist.return_value = [.9]
            result.keypoints.data.cpu.return_value.tolist.return_value = [[joint] * 16 + [[0,0,0]]]
            return [result]

        detector, pose = Mock(), Mock()
        detector.predict.side_effect = [detected([30,30,60,80]), detected([66,72,96,132])]
        # Second target in original coordinates: [450,200,480,260].
        # Its 256-square pose crop starts at (337,102).
        pose.predict.side_effect = [posed([30,30,60,80], [40,50,.9]), posed([113,98,143,158], [123,128,.9])]
        with patch('pose_inference.tile_windows', return_value=[(384,128,640,384)]):
            result = predict_person_zoom(detector, pose, picture, 'cpu', .15, .45)
        points = [p['keypoints'] for p in result['people']]
        self.assertCountEqual([p[0] for p in points], [[40,50,2], [460,230,2]])
        self.assertTrue(all(p[-1] == [0,0,0] for p in points))
        self.assertEqual(result['passes'], 4)

    def test_tiles_cover_frame_with_overlap_and_bounded_cost(self):
        windows = tile_windows(1280, 800)
        self.assertEqual(len(windows), 6)
        for x in range(0, 1280, 13):
            for y in range(0, 800, 13):
                self.assertTrue(any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in windows))
        self.assertTrue(all(x1-x0 == 512 and y1-y0 == 512 for x0,y0,x1,y1 in windows))
        self.assertLessEqual(len(tile_windows(800, 800)), 9)
        self.assertEqual(tile_windows(64, 64), [])

    def test_mapping_preserves_missing_joints_and_original_pixels(self):
        data, boxes, scores = Mock(), Mock(), Mock()
        data.cpu.return_value.tolist.return_value = [[[12, 30, .9], [0, 0, .9], [20, 20, .1]] + [[25, 40, .8]] * 14]
        boxes.cpu.return_value.tolist.return_value = [[10, 20, 50, 80]]
        scores.cpu.return_value.tolist.return_value = [.8]
        result = SimpleNamespace(keypoints=SimpleNamespace(data=data), boxes=SimpleNamespace(xyxy=boxes, conf=scores))
        candidates = result_candidates(result, (384, 288), (1280, 800), .45, 4)
        self.assertEqual(candidates[0]['keypoints'][0], [396, 318, 2])
        self.assertEqual(candidates[0]['keypoints'][1:3], [[0,0,0], [0,0,0]])
        self.assertEqual(candidates[0]['box'], [394,308,434,368])
        self.assertEqual(candidates[0]['visible'], 15)
        self.assertEqual(result_candidates(result, (0,0), (1280,800), .95, 4), [])

    def test_merge_keeps_best_duplicate_and_distinct_nearby_person(self):
        weak = {'box': [100,100,150,200], 'score': .5, 'visible': 10, 'keypoints': []}
        strong = {'box': [102,102,152,202], 'score': .9, 'visible': 14, 'keypoints': []}
        nearby = {'box': [145,100,195,200], 'score': .8, 'visible': 12, 'keypoints': []}
        self.assertEqual(merge_candidates([weak, strong, nearby]), [strong, nearby])
        self.assertEqual(box_iou([0,0,0,0], [0,0,0,0]), 0)

    def test_weak_candidates_need_joints_and_independent_views(self):
        weak = {'box': [100,100,150,200], 'score': .3, 'visible': 12, 'view_id': 0}
        self.assertEqual(merge_candidates([weak]), [])
        self.assertEqual(merge_candidates([weak, dict(weak)]), [])
        second = {**weak, 'view_id': 1}
        self.assertEqual(len(merge_candidates([weak, second])), 1)
        self.assertEqual(merge_candidates([{**weak, 'visible': 10}, {**second, 'visible': 10}]), [])


if __name__ == '__main__':
    unittest.main()
