"""Source-registry models: the configured corpus."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import SourceType
from app.models.retrieval import SourceConfig


class SourceRegistryFile(BaseModel):
    """Schema of ``sources.json``."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"
    description: str = ""
    amc: str = "HDFC Mutual Fund"
    sourceTypes: dict[str, str] = Field(default_factory=dict)  # noqa: N815 - source file key
    schemes: list[SchemeRecord] = Field(default_factory=list)
    sources: list[SourceConfig] = Field(default_factory=list)


class SchemeRecord(BaseModel):
    """A scheme derived from the source registry, served to the frontend."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    amc: str
    plan: str = "Direct Growth"
    categories: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    @property
    def authority_source_type(self) -> SourceType:
        return SourceType.AMC_OFFICIAL
