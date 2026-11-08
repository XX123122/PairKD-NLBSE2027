"""Verify repository payloads and frozen identity without importing a model."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.source_guard import verify_release

if __name__=='__main__':
    print(json.dumps(verify_release()))
