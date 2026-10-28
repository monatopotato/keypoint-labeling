"""Run from the project root: python -m unittest discover -s tests -v"""
import csv
import io
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

# Set a fresh location before the application's module-level constants are defined.
_TEST_TMP = tempfile.TemporaryDirectory()
os.environ["POSEFRAME_DATA"] = _TEST_TMP.name
from app import app, filename_sort_key, KEYPOINTS, CSV_FIELDS  # noqa: E402


class PoseFrameTests(unittest.TestCase):
    def setUp(self):
        self.client=app.test_client()
        self.folder=Path(_TEST_TMP.name)/'test_photos'
        self.folder.mkdir(exist_ok=True)
        for old in self.folder.iterdir():
            if old.is_file(): old.unlink()
        for name in ['2026-10-21-12-30-02-100.jpg', '2026-10-21-12-30-01-000.jpg', '2026-10-21-12-30-02-000.jpg']:
            (self.folder/name).write_bytes(b'fake-image-bytes')
        response=self.client.post('/api/import/path',json={'path':str(self.folder)})
        self.assertEqual(response.status_code,200)

    def test_sorting(self):
        expected=['frame_2.png','frame_10.png']
        self.assertEqual(sorted(expected, key=filename_sort_key),expected)
        frames=self.client.get('/api/project').json['frames']
        self.assertEqual(frames[0]['name'],'2026-10-21-12-30-01-000.jpg')
        self.assertEqual(frames[1]['name'],'2026-10-21-12-30-02-000.jpg')

    def test_save_and_export_multiple_people(self):
        name='2026-10-21-12-30-01-000.jpg'
        people=[{'source':'manual','keypoints':[[float(i*3),float(i*4),2] for i in range(17)]},
                {'source':'yolo26','keypoints':[[0,0,0] for i in range(17)]}]
        response=self.client.put('/api/annotations/'+name,json={'width':640,'height':480,'people':people})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['people'],2)
        self.assertEqual(self.client.get('/api/annotations/'+name).json['people'][0]['keypoints'][4],[12.0,16.0,2])
        rows=list(csv.DictReader(io.StringIO(self.client.get('/api/export').data.decode('utf-8'))))
        self.assertEqual(len(rows),4)  # two people on first frame, one placeholder per other frame
        self.assertEqual(len(rows[0]),59)
        self.assertEqual(rows[0]['person_id'],'1')
        self.assertEqual(rows[0]['right_ankle_v'],'2')
        self.assertEqual(rows[1]['source'],'yolo26')
        self.assertEqual(rows[1]['num_keypoints'],'0')
        self.assertEqual(rows[-1]['annotation_status'],'unlabeled')

    def test_visibility_zero_forces_zero_coordinates(self):
        name='2026-10-21-12-30-02-000.jpg'
        points=[[13,14,0]]+[[23,45,1] for _ in range(16)]
        response=self.client.put('/api/annotations/'+name,json={'width':640,'height':480,'people':[{'keypoints':points}]})
        self.assertEqual(response.status_code,200)
        result=self.client.get('/api/annotations/'+name).json
        self.assertEqual(result['people'][0]['keypoints'][0],[0,0,0])
        self.assertEqual(result['people'][0]['keypoints'][1],[23.0,45.0,1])

    def test_zero_people_and_validation(self):
        name='2026-10-21-12-30-01-000.jpg'
        self.assertEqual(self.client.put('/api/annotations/'+name,json={'width':100,'height':100,'people':[]}).json['status'],'reviewed_empty')
        invalid=self.client.put('/api/annotations/'+name,json={'width':100,'height':100,'people':[{'keypoints':[[0,0,2]]}]})
        self.assertEqual(invalid.status_code,400)

    def test_upload_and_missing_yolo(self):
        self.assertEqual(self.client.post('/api/import/start').status_code,200)
        data={'files':[(io.BytesIO(b'example-image'), 'frame_01.jpg')]}
        self.assertEqual(self.client.post('/api/import/batch',data=data,content_type='multipart/form-data').json['saved'],1)
        self.assertEqual(self.client.post('/api/import/finish').json['count'],1)
        self.assertEqual(self.client.get('/api/image/frame_01.jpg').status_code,200)
        self.assertEqual(self.client.get('/api/image/not_found.jpg').status_code,404)

    def test_predict_reports_windows_dependency_failure(self):
        error = OSError('Error loading c10.dll or one of its dependencies')
        error.winerror = 126
        with patch('app.importlib.util.find_spec', return_value=object()), patch('app.load_yolo_model', side_effect=error):
            response = self.client.post('/api/predict/2026-10-21-12-30-01-000.jpg', json={})
        self.assertEqual(response.status_code, 503)
        self.assertIn('Visual C++ x64 runtime', response.json['error'])
        self.assertIn('restart run.bat', response.json['error'])

    def test_predict_converts_confidence_to_visibility(self):
        result = Mock()
        result.keypoints.data.cpu.return_value.tolist.return_value = [
            [[10, 20, 0.9], [30, 40, 0.2]] + [[50, 60, 0.8] for _ in range(15)]
        ]
        model = Mock()
        model.predict.return_value = [result]
        with patch('app.importlib.util.find_spec', return_value=object()), patch('app.load_yolo_model', return_value=model):
            response = self.client.post('/api/predict/2026-10-21-12-30-01-000.jpg', json={})
        self.assertEqual(response.status_code, 200)
        points = response.json['people'][0]['keypoints']
        self.assertEqual(len(points), 17)
        self.assertEqual(points[0], [10, 20, 2])
        self.assertEqual(points[1], [0, 0, 0])


if __name__=='__main__': unittest.main()
