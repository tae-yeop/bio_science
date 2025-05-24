from typing import Literal

from pydantic import BaseModel


TASK = Literal[
    "finetune",
    "pretrain"
]
class Config(BaseModel):
    task: TASK