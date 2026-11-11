import json
from unittest.mock import patch

import numpy as np
import torch
from django.test import TestCase

from views import CLASS_IDS


class PredictionApiTests(TestCase):
    def test_index_renders_demo(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "KSL Sign Recognition")

    def test_predict_accepts_expected_batch_shape(self):
        sequence = np.zeros((1, 200, 450), dtype=np.float32).tolist()
        response = self.client.post(
            "/api/predict/",
            data=json.dumps({"sequence": sequence}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn(result["class_id"], CLASS_IDS)
        self.assertIn("class_index", result)
        self.assertIn("word", result)
        self.assertIn("confidence", result)

    def test_predict_returns_gloss_when_available(self):
        class_index = 7
        logits = torch.zeros((1, len(CLASS_IDS)))
        logits[0, class_index] = 5
        sequence = np.zeros((1, 200, 450), dtype=np.float32).tolist()
        with (
            patch("views.MODEL", return_value=logits),
            patch("views.KOREAN_GLOSSES", {CLASS_IDS[class_index]: "안녕하세요"}),
        ):
            response = self.client.post(
                "/api/predict/",
                data=json.dumps({"sequence": sequence}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["word"], "안녕하세요")
        self.assertEqual(response.json()["korean_word"], "안녕하세요")

    def test_predict_rejects_wrong_shape(self):
        response = self.client.post(
            "/api/predict/",
            data=json.dumps({"sequence": [[[0.0] * 450] * 199]}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Expected sequence shape", response.json()["error"])
