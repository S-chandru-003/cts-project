import os
import sys
import unittest
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app, THRESHOLD, DATA_DICTIONARY
from src.components.feature_engineering import FeatureEngineering

client = TestClient(app)

class TestHealthcareFraudSystem(unittest.TestCase):
    
    def test_health_endpoint(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["threshold"], THRESHOLD)

    def test_data_dictionary_endpoint(self):
        response = client.get("/data-dictionary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tables", data)
        self.assertIn("PROVIDERS", data["tables"])
        self.assertIn("BENEFICIARY", data["tables"])
        self.assertIn("INPATIENT", data["tables"])
        self.assertIn("OUTPATIENT", data["tables"])
        self.assertIn("relationships", data)
        self.assertEqual(len(data["relationships"]), 4)

    def test_predict_sample_and_dashboard(self):
        # 1. Run sample prediction
        pred_res = client.post("/predict/sample")
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.json()
        
        self.assertTrue(pred_data["success"])
        analysis_id = pred_data["analysis_id"]
        self.assertTrue(bool(analysis_id))
        self.assertIn("summary", pred_data)
        self.assertGreater(pred_data["summary"]["total_providers"], 0)
        
        # 2. Check Dashboard Summary with Geographic and Peer Insights
        dash_res = client.get(f"/dashboard?analysis_id={analysis_id}")
        self.assertEqual(dash_res.status_code, 200)
        dash_data = dash_res.json()
        self.assertIn("peer_benchmarks", dash_data)
        self.assertIn("geographic_insights", dash_data)
        self.assertIn("claim_type_summary", dash_data)
        self.assertIn("top_risk_providers", dash_data)
        
        # 3. Check Provider Deep Dive
        top_provider = dash_data["top_risk_providers"][0]["Provider"]
        prov_res = client.get(f"/provider/{top_provider}?analysis_id={analysis_id}")
        self.assertEqual(prov_res.status_code, 200)
        prov_data = prov_res.json()
        
        self.assertIn("why_flagged", prov_data)
        self.assertIn("human_readable_reasons", prov_data["why_flagged"])
        self.assertGreater(len(prov_data["why_flagged"]["human_readable_reasons"]), 0)
        self.assertIn("statistics", prov_data)
        self.assertIn("claim_anomalies", prov_data)
        self.assertIn("charts", prov_data)

        # 4. Check Provider Comparison Endpoint
        if len(dash_data["top_risk_providers"]) > 1:
            p2 = dash_data["top_risk_providers"][1]["Provider"]
            comp_res = client.get(f"/compare?provider1={top_provider}&provider2={p2}&analysis_id={analysis_id}")
            self.assertEqual(comp_res.status_code, 200)
            comp_data = comp_res.json()
            self.assertIn("provider1", comp_data)
            self.assertIn("provider2", comp_data)
            self.assertIn("peer_benchmarks", comp_data)

if __name__ == "__main__":
    unittest.main()
