import pytest
from pathlib import Path
from pipelines.data_pipeline import _download_heart_disease_dataset

def test_data_pipeline_download_step_creates_raw_file(tmp_path):
    target_path = tmp_path / "heart_disease.csv"
    frame = _download_heart_disease_dataset(target_path=target_path)

    assert target_path.exists()
    assert not frame.empty
    assert "target" in frame.columns