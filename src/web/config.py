"""单机后端配置及预登记输入清单。"""

from dataclasses import dataclass, field
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Settings:
    database: Path = field(default_factory=lambda: Path(os.environ.get("LOG_APP_DATABASE", ROOT / "runtime" / "app.db")))
    input_dir: Path = ROOT / "datasets" / "processed"
    manifest_key: str = "openstack_processed_v1"
    expected_logs: int | None = 207820
    expected_instances: int | None = 2069
    expected_templates: int | None = 130

    def manifest(self):
        return {name: self.input_dir / name for name in (
            "openstack_structured.csv", "openstack_template_analysis.csv",
            "openstack_template_statistics.csv", "openstack_instance_labels.csv")}
