from sqlmodel import SQLModel, Field, Relationship
from typing import Optional

class User(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    name: str = Field(index=True)
    password: str = Field(max_length=256, min_length=1)
    dataset: Optional['Dataset'] = Relationship(back_populates='user')
    mlmodel: Optional['MLModel'] = Relationship(back_populates='user')
    predictions: Optional['Prediction'] = Relationship(back_populates='user')

class UserInput(SQLModel):
    name: str
    password: str

class UserLogin(SQLModel):
    name: str
    password: str
