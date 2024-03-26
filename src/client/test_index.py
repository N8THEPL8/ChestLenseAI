import unittest
import index

class TestIndex(unittest.TestCase):
    def setUp(self):
        self.app = index.app.test_client()

    def test_fetch_image(self):
        response = self.app.get('/fetchimage')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"filename": "image.jpg"})

    def test_delete_image(self):
        response = self.app.get('/deleteimage')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"true": True})

if __name__ == '__main__':
    unittest.main()