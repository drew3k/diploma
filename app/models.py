from pydantic import BaseModel
from typing import List, Optional


class ProcessRequest(BaseModel):
    policy: str = "mask"  # or "remove"
    languages: List[str] = ["ru", "en"]
    types: Optional[List[str]] = (
        None  # e.g. ["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","IP_ADDRESS","CREDIT_CARD"]
    )


class ProcessResult(BaseModel):
    input_name: str
    output_name: str
    output_url: str
    found: int
    filetype: str
