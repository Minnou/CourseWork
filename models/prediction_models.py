from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship

class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: int = Field(foreign_key="mlmodel.id")
    input_dataset_id: int = Field(foreign_key="dataset.id")
    owner_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    result_file_path: str
    user: Optional['User'] = Relationship(back_populates='predictions')
    dataset: Optional['Dataset'] = Relationship(back_populates='predictions')
    mlmodel: Optional['MLModel'] = Relationship(back_populates='predictions')