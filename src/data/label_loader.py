"""读取压缩包内的实例级异常标签。"""

import re
import tarfile

from .log_parser import UUID_PATTERN


def load_anomaly_uuids(archive: tarfile.TarFile) -> set[str]:
    member = archive.getmember("anomaly_labels.txt")
    if not member.isfile():
        raise ValueError("anomaly_labels.txt 必须是普通文件")
    labels = set()
    with archive.extractfile(member) as stream:
        for raw_line in stream:
            line = raw_line.decode("utf-8-sig").strip()
            if re.fullmatch(UUID_PATTERN, line, flags=re.IGNORECASE):
                labels.add(line.lower())
    if not labels:
        raise ValueError("anomaly_labels.txt 缺少有效的实例 UUID")
    return labels
