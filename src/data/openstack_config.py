"""OpenStack 数据集默认路径和文件列表。"""

from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"
ARCHIVE_PATH = DATASET_ROOT / "raw" / "OpenStack.tar.gz"
OUTPUT_DIR = DATASET_ROOT / "processed"
LOG_FILES = ("openstack_abnormal.log", "openstack_normal1.log", "openstack_normal2.log")
