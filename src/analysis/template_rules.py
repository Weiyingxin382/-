"""基于 OpenStack 消息格式的规则模板基线。"""

import re
from ipaddress import IPv4Address, AddressValueError

from src.data.log_parser import UUID_PATTERN


# 优先匹配完整的版本、状态码及标识符，避免其数字片段被通用规则处理。
TOKENS = re.compile(
    rf"(?P<request_id>\breq-{UUID_PATTERN}\b)"
    rf"|(?P<uuid>(?<=/instances/){UUID_PATTERN}(?=_del\b)|(?<![\w-]){UUID_PATTERN}(?![\w-]))"
    r"|(?P<hex_id>\b(?:[0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{32})\b)"
    r"|(?P<hostname>(?<=host ')[a-z0-9-]+(?:\.[a-z0-9-]+)+(?='))"
    r"|(?P<static_number>\bHTTP/\d+(?:\.\d+)?\b(?:[\"']?\s+[1-5]\d{2}\b)?"
    r"|\bstatus\s*[:=]\s*[1-5]\d{2}\b|/openstack/\d{4}-\d{2}-\d{2}(?=/|\b)"
    r"|/flavors/\d+\b|\bpython\d+\.\d+\b)"
    r"|(?P<timestamp>(?<=changes-since=)\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}"
    r"(?:\.\d+)?(?:Z|(?:%2B|-)\d{2}%3A\d{2})?(?=[&#\s\"']|$)"
    r"|(?<![\w/])\d{4}-\d{2}-\d{2}(?:[ T_]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?(?!\w)"
    r"|(?<![\w:])\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?![\w:]))"
    r"|(?P<ipv4>(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\d))"
    r"|(?P<number>(?<![\w.])[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
    r"(?!\w|\.\d))",
    re.IGNORECASE,
)
FILE_PATH = re.compile(r"(?<![\w/])/(?:var|etc|opt|tmp|usr|dev|run|home|root|srv|mnt|lib|bin|proc|sys)/[^\s\"'<>(),;:\[\]]+")
API_PATH = re.compile(r"(?<![\w/])/(?:v\d+(?:\.\d+)?/|openstack/|latest/)[^\s\"'<>]+")
PLACEHOLDERS = {
    "request_id": "<REQ_ID>", "uuid": "<UUID>", "hex_id": "<HEX_ID>",
    "hostname": "<HOST>", "timestamp": "<TIMESTAMP>", "ipv4": "<IPV4>",
    "number": "<NUM>",
}

RULE_NOTES = [
    "模板键只使用归一化后的 message。原始字段保留供追溯。",
    "分别替换请求 ID、标准 UUID、32/40/64 位十六进制标识符及 host 单引号内的域名。",
    "将有效 IPv4 和日期时间替换为占位符，覆盖 changes-since 参数中的 URL 编码时间。保留 HTTP 版本、显式状态码、元数据 API 日期版本和数字 flavor ID。",
    "独立整数、小数、科学计数法替换为 <NUM>，保留旁边的单位及指标名。嵌入词中的数字保留。",
    "文件路径保留静态目录和文件名，仅替换其中匹配到的动态标识符或数值。实例删除目录 /instances/<UUID>_del 保留 _del 后缀，API 路由保留静态资源段。",
    "参数分布统计按包含该类型的日志行数计。路径与其嵌入的标识符可能在同一行分别计数。",
]
LIMITATIONS = [
    "规则依赖消息写法。未覆盖 IPv6、非标准标识符和任意动态文件名，等义消息可能分裂。",
    "日期和时间按文本格式识别，可能匹配形似日期的业务值；稳定目录中的独立数字也可能被替换。",
    "数值大小和十六进制标识符类型会被合并，因此同一模板可以对应不同业务结果。",
    "请求上下文结构保留，带请求上下文和空上下文的消息可能产生不同模板。",
    "异常实例统计使用结构化 CSV 中可关联的 UUID，无法推断其他关联日志或逐条异常状态。",
]


def normalize_message(message: str) -> tuple[str, set[str]]:
    """返回归一化正文以及原文中识别到的参数类型。"""
    parameters = set()
    if FILE_PATH.search(message):
        parameters.add("file_path")
    if API_PATH.search(message):
        parameters.add("api_path")

    def replace(match: re.Match) -> str:
        kind = match.lastgroup
        original = match.group()
        if kind == "static_number":
            return original
        if kind == "ipv4":
            try:
                IPv4Address(original)
            except AddressValueError:
                return original
        parameters.add(kind)
        return PLACEHOLDERS[kind]

    return TOKENS.sub(replace, message).strip(), parameters
