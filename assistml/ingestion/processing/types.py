from typing import Dict, Optional

from pydantic import BaseModel

from common.data.task import TaskType


class ProcessingOptions(BaseModel):
    head: Optional[int] = None
    recursive: bool = False
    offset: Optional[Dict[str, int]] = None
    task_type: Optional[TaskType] = None
