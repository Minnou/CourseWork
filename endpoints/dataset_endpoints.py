from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from datetime import datetime
from models.dataset_models import Dataset
from models.user_models import User
from db.db import session
from repos.dataset_repository import create_dataset, valutes, read_dataset, group_dataset_by_month
from auth.auth import AuthHandler
import os
from typing import List

dataset_router = APIRouter(prefix="/datasets", tags=["Datasets"])
auth_handler = AuthHandler()

@dataset_router.post("/create")
async def create_dataset_endpoint(valute: str, days: int, user: User = Depends(auth_handler.get_current_user)):
    if valute.upper() not in valutes:
        raise HTTPException(status_code=400, detail="Нcurrent_userедопустимый код валюты")

    try:
        file_path = create_dataset(valute.upper(), days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {e}")

    dataset = Dataset(
        name=f"{valute.upper()}_{days}_days",
        owner_id=user.id,
        created_at=datetime.utcnow(),
        file_path=file_path
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset

@dataset_router.get("/{dataset_id}")
async def get_dataset(dataset_id: int, user: User = Depends(auth_handler.get_current_user)):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if (dataset.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return dataset

@dataset_router.get("/user/")
async def get_user_datasets(user: User = Depends(auth_handler.get_current_user)):
    statement = select(Dataset).where(Dataset.owner_id == user.id)
    datasets = session.exec(statement).all()
    return datasets

@dataset_router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, user: User = Depends(auth_handler.get_current_user)):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if (dataset.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Удаление файла с диска
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

    session.delete(dataset)
    session.commit()
    return {"message": "Dataset deleted successfully"}

@dataset_router.get("/{dataset_id}/download")
async def download_dataset(dataset_id: int, user: User = Depends(auth_handler.get_current_user)):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if (dataset.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(dataset.file_path, media_type="application/octet-stream", filename=os.path.basename(dataset.file_path))

@dataset_router.get("/{dataset_id}/data")
async def get_dataset_data(dataset_id: int, user: User = Depends(auth_handler.get_current_user)):
    dataset = session.exec(select(Dataset).where(Dataset.id == dataset_id)).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if (dataset.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")

    return read_dataset(dataset.file_path)

@dataset_router.post("/group-by-month/{dataset_id}")
def group_by_month(dataset_id: int, user: User = Depends(auth_handler.get_current_user)):
    original_dataset = session.exec(select(Dataset).where(Dataset.id == dataset_id)).first()
    if not original_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    new_path = group_dataset_by_month(original_dataset.file_path)

    new_dataset = Dataset(
        name=f"{original_dataset.name}_monthly",
        owner_id=user.id,
        created_at=datetime.utcnow(),
        file_path=new_path
    )
    session.add(new_dataset)
    session.commit()
    session.refresh(new_dataset)

    return new_dataset