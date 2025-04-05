from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from datetime import datetime
from models.dataset_models import Dataset
from models.prediction_models import Prediction
from db.db import session
from auth.auth import AuthHandler

prediction_router = APIRouter(prefix="/predictions", tags=["Predictions"])
auth_handler = AuthHandler()