from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from datetime import datetime
from models.dataset_models import Dataset
from models.ml_models import MLModel
from db.db import session
from auth.auth import AuthHandler

ml_router = APIRouter(prefix="/mls", tags=["ML"])
auth_handler = AuthHandler()

@ml_router.get("/get")
def get():
    pass