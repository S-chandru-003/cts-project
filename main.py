import os
import sys
from pathlib import Path

# Add the project directory to sys.path
PROJECT_DIR = Path(__file__).resolve().parent / "Healthcare-Insurance-claim-fraud-detection-main"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Expose FastAPI app for root-level uvicorn runners
from backend.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
