from dataclasses import dataclass
import os

@dataclass
class Paths:
    root: str = "."
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    text_dir: str = os.path.join("models", "text")
    fusion_dir: str = os.path.join("models", "fusion")
    fl_dir: str = os.path.join("models", "fl")
    results_file: str = os.path.join("models", "results.json")

DATASETS = {
    "FakeNews": "processed_FakeNews.csv",
    "LIAR": "processed_LIAR.csv",
}

