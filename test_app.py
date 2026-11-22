import unittest

from app import app


class AppRoutesTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_homepage_is_served_at_root(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"House Price Estimator", response.data)
        self.assertNotIn(b'"status":"ok"', response.data)

    def test_health_endpoint_remains_available(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertGreater(response.json["trained_rows"], 0)

    def test_model_info_endpoint_keeps_working(self):
        response = self.client.get("/model-info")

        self.assertEqual(response.status_code, 200)
        self.assertIn("feature_order", response.json)

    def test_predict_endpoint_keeps_working(self):
        response = self.client.post("/predict", json={
            "income": 65000,
            "age": 5,
            "rooms": 6,
            "bedrooms": 3,
            "population": 35000,
        })

        self.assertEqual(response.status_code, 200)
        self.assertIn("predicted_price", response.json)


if __name__ == "__main__":
    unittest.main()
