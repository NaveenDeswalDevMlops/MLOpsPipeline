import importlib
from pathlib import Path

def test_eda_image_discovery_uses_generated_pipeline_filenames(monkeypatch):
    workspace = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("WORKSPACE", str(workspace))

    import dashboard.config as dashboard_config
    import dashboard.services as dashboard_services

    importlib.reload(dashboard_config)
    importlib.reload(dashboard_services)

    available = dashboard_services.get_eda_images()
    names = {path.name for _, path in available}

    expected_files = {
        "correlation_heatmap.png", "boxplot.png", "countplot.png", 
        "class_balance.png", "feature_distribution.png", "feature_importance.png"
    }
    
    assert expected_files.issubset(names)
    assert any(name.startswith("histogram_") for name in names)