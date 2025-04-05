from typing import Optional, List
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Relationship

class MLModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    dataset_id: int = Field(foreign_key="dataset.id")
    owner_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_path: str
    user: Optional['User'] = Relationship(back_populates='mlmodel')
    dataset: Optional['Dataset'] = Relationship(back_populates='mlmodel')
    predictions: Optional['Prediction'] = Relationship(back_populates='mlmodel')