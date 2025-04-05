from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field,  Relationship

class Dataset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    owner_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_path: str
    user: Optional['User'] = Relationship(back_populates='dataset')
    mlmodel: Optional['MLModel'] = Relationship(back_populates='dataset')
    predictions: Optional['Prediction'] = Relationship(back_populates='dataset')
