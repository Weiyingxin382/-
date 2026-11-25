"""解析 OpenStack 日志头部和 VM 实例引用。"""

import re


UUID_PATTERN = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
HEADER = re.compile(
    r"(?P<component>[\w-]+)\.log(?:\.\d+)?\.\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}"
    r"\s+(?P<event_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{1,6})"
    r"\s+(?P<process_id>\d+)\s+(?P<severity>[A-Z]+)\s+(?P<module>[\w.]+)"
    r"(?:[ \t](?P<message>.*))?"
)
INSTANCE = re.compile(
    rf"(?:\[instance:\s*|\b(?:for )?instance\s+|/(?:servers|instances)/)"
    rf"({UUID_PATTERN})(?![\w-])",
    re.IGNORECASE,
)

PARSED_FIELDS = (
    "component", "event_time", "process_id", "severity", "module", "message",
    "instance_uuid",
)


def parse_log_line(raw_text: str) -> dict | None:
    match = HEADER.fullmatch(raw_text)
    if match is None:
        return None
    fields = match.groupdict()
    fields["process_id"] = int(fields["process_id"])
    fields["message"] = fields["message"] or ""
    # 请求 ID、镜像 ID 等也使用 UUID，只从明确指向 VM 的上下文提取。
    candidates = {uuid.lower() for uuid in INSTANCE.findall(fields["message"])}
    fields["instance_uuid"] = candidates.pop() if len(candidates) == 1 else ""
    return fields
