
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class ProjectConfig:

    project_root: Path

    def __post_init__(self):
        self.project_root = Path(
            self.project_root
        ).expanduser().resolve()

        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.src_dir = self.project_root / "src"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.reports_dir = self.project_root / "reports"
        self.dashboard_dir = self.project_root / "dashboard"
        self.cache_dir = self.project_root / "cache"

        self.required_directories = [
            self.raw_dir,
            self.processed_dir,
            self.src_dir,
            self.models_dir,
            self.results_dir,
            self.reports_dir,
            self.dashboard_dir,
            self.cache_dir
        ]

    def create_directories(self):
        for directory in self.required_directories:
            directory.mkdir(
                parents=True,
                exist_ok=True
            )

    def get_path(self, relative_path):
        return self.project_root / relative_path

    def validate(self):
        self.create_directories()

        return {
            "project_root_exists": self.project_root.exists(),
            "raw_directory_exists": self.raw_dir.exists(),
            "processed_directory_exists": self.processed_dir.exists(),
            "models_directory_exists": self.models_dir.exists(),
            "results_directory_exists": self.results_dir.exists()
        }

    def save(self):
        data = {
            "project_root": str(self.project_root),
            "raw_dir": str(self.raw_dir),
            "processed_dir": str(self.processed_dir),
            "models_dir": str(self.models_dir),
            "results_dir": str(self.results_dir)
        }

        with open(
            self.project_root / "project_config.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(data, file, indent=4)


def load_project_config():
    config = ProjectConfig(
        Path.home() / "Downloads" / "Uttarakhand"
    )

    config.create_directories()
    config.save()

    return config


if __name__ == "__main__":
    config = load_project_config()
    print(config.validate())
