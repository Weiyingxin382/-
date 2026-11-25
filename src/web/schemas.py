"""请求参数校验；不带时区的原日志时间单独校验。"""

from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.web.importer import normalize_time


class LogQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: int = Field(gt=0)
    q: str | None = Field(default=None, max_length=200)
    start_time: str | None = None
    end_time: str | None = None
    severity: Literal["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL", "TRACE"] | None = None
    component: str | None = Field(default=None, max_length=100)
    module: str | None = Field(default=None, max_length=200)
    source_id: int | None = Field(default=None, gt=0)
    instance_uuid: UUID | None = None
    template_id: str | None = Field(default=None, max_length=100)
    page: int | None = Field(default=None, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=2000)
    include_total: bool = True
    bucket: Literal["hour", "day"] = "hour"

    @model_validator(mode="after")
    def validate_bounds(self):
        for field in ("start_time", "end_time"):
            value = getattr(self, field)
            if value:
                setattr(self, field, normalize_time(value))
        if self.start_time and self.end_time and self.start_time >= self.end_time:
            raise ValueError("start_time 必须早于 end_time")
        if self.cursor and self.page is not None:
            raise ValueError("cursor 与 page 不能同时传入")
        return self


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_manifest_key: str = Field(max_length=100)
    dataset_name: str = Field(min_length=1, max_length=100)
