import unittest
import index
import json
from index import Patient, NewScan

class TestIndex(unittest.TestCase):
    def setUp(self):
        self.app = index.app.test_client()
        self.app.testing = True

    def test_fetch_image(self):
        response = self.app.get('/fetchimage')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"filename": "image.jpg"})

    def test_delete_image(self):
        response = self.app.get('/deleteimage')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"true": True})
  
    # Testing for login() or base route ('/')
    def test_login_get_request(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    def test_login_post_request_existing_doctor(self):
        response =  self.app.post('/', data={'username': 'doctor3@gmail.com', 'password': 'password3'})
        self.assertEqual(response.status_code, 302) 

    def test_login_post_request_non_existing_doctor(self):
        response =  self.app.post('/', data={'username': 'doctor2@gmail.com', 'password': 'password3'})
        self.assertEqual(response.status_code, 200) 
        self.assertNotIn(b'Invalid username or password', response.data)

    # Testing for doctor() or route ('/doctor')
    def test_doctor_authenticated(self):
        with self.app as client:
            with client.session_transaction() as sess:
                sess['d_id'] = 3
            response = client.get('/doctor')
            doctor_id = "d3"
            patients = Patient.query.filter_by(d_id=doctor_id).all()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'text/html; charset=utf-8')
            self.assertTrue(patients)
            self.assertIn(b'List of Patients', response.data)

    def test_doctor_unauthenticated(self):
        with self.app as client:
            response = client.get('/doctor')
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.location, '/')

    # Testing for comments() or route ('/comments')
    def test_comments_update_comment(self):
        with self.app as client:
            response = client.post('/comments', json={'scan_id': '50712315', 'textarea_content': 'Updated comment'})
            scan = NewScan.query.filter_by(s_id='50712315').first()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'application/json')
            self.assertIsNotNone(scan)
            self.assertEqual(scan.s_comment, 'Updated comment')

    def test_comments_missing_data(self):
        with self.app as client:
            response = client.post('/comments', json={})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'application/json')
            response_data = response.get_json()
            self.assertIn('Error', response_data)

    # Testing for index(patient_id) or route ('/index/<patient_id>')
    def test_index_patient_found(self):
        with self.app as client:
            response = client.get('/index/17007063')
            patient = Patient.query.get('17007063')
            scans = NewScan.query.filter_by(p_id='17007063').all()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'text/html; charset=utf-8')
            self.assertTrue(patient)
            self.assertTrue(scans)
            self.assertIn(b'Patient Name', response.data)
            self.assertIn(b'Select Previous Scans', response.data)  

    def test_index_patient_not_found(self):
        with self.app as client:
            response = client.get('/index/999')
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.location, '/doctor')

    # Testing for upload() or route ('/upload')
    def test_upload_new_image(self):
        with open('../demo_images/Steven.dcm', 'rb') as img:
            data = {
                'xrayImage': (img, 'Steven.dcm'),
                'uploadedImages': '17007063'
            }
            response = self.app.post('/upload', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['Study_ID'], '50712315')
        self.assertEqual(response_data['Patient_ID'], '17007063')
        self.assertEqual(response_data['Patient_Name'], 'Steven Sanders')
        self.assertEqual(response_data['Patient_Sex'], 'M')
        self.assertEqual(response_data['Patient_Birth_Date'], '1989-01-19')
        self.assertEqual(response_data['Acquisition_Date'], '2009-11-01')
        self.assertEqual(response_data['View_Position'], 'AP')
        self.assertEqual(response_data['Patient_Age_at_Time_of_Acquisition'], '20 years, 9 months, 13 days')
        self.assertEqual(response_data['comment'], 'Updated comment')

    # Testing for upload_our_model() or route ('/upload-our-model')
    def test_upload_our_model_new_image(self):
        with open('../demo_images/Steven.dcm', 'rb') as img:
            data = {
                'xrayImage': (img, 'Steven.dcm'),
                'uploadedImages': '17007063'
            }
            response = self.app.post('/upload-our-model', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['Study_ID'], '50712315')
        self.assertEqual(response_data['Patient_ID'], '17007063')
        self.assertEqual(response_data['Patient_Name'], 'Steven Sanders')
        self.assertEqual(response_data['Patient_Sex'], 'M')
        self.assertEqual(response_data['Patient_Birth_Date'], '1989-01-19')
        self.assertEqual(response_data['Acquisition_Date'], '2009-11-01')
        self.assertEqual(response_data['View_Position'], 'AP')
        self.assertEqual(response_data['Patient_Age_at_Time_of_Acquisition'], '20 years, 9 months, 13 days')
        self.assertEqual(response_data['comment'], 'Updated comment')

if __name__ == '__main__':
    unittest.main()