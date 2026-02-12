from pydantic import BaseModel, ConfigDict


class TurnBaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TurnText(TurnBaseModel):
    body: str


class TurnMessage(TurnBaseModel):
    id: str
    type: str
    text: TurnText | None = None


class TurnWebhook(TurnBaseModel):
    messages: list[TurnMessage]
