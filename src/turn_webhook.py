from pydantic import BaseModel, ConfigDict, Field


class TurnBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TurnText(TurnBaseModel):
    body: str


class TurnMessage(TurnBaseModel):
    id: str
    type: str
    text: TurnText | None = None


class TurnWebhook(TurnBaseModel):
    messages: list[TurnMessage] = Field(default_factory=list)
    statuses: list[dict] = Field(default_factory=list)
