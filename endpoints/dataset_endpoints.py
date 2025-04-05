from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select
from datetime import datetime
from models.dataset_models import Dataset
from models.user_models import User
from db.db import session
from repos.get_valute import create_dataset, valutes
from auth.auth import AuthHandler
import os
from typing import List

dataset_router = APIRouter(prefix="/datasets", tags=["Datasets"])
auth_handler = AuthHandler()

@dataset_router.post("/create")
async def create_dataset_endpoint(
    valute: str,
    days: int,
    current_user: User = Depends(auth_handler.get_current_user)
):
    if valute.upper() not in valutes:
        raise HTTPException(status_code=400, detail="Недопустимый код валюты")

    try:
        file_path = create_dataset(valute.upper(), days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {e}")

    dataset = Dataset(
        name=f"{valute.upper()}_{days}_days",
        owner_id=current_user.id,
        created_at=datetime.utcnow(),
        file_path=file_path
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


# Получение одного датасета по ID
@dataset_router.get("/datasets/{dataset_id}", response_model=Dataset)
def get_dataset(dataset_id: int):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

# Получение всех датасетов пользователя
@dataset_router.get("/datasets/user/{user_id}", response_model=List[Dataset])
def get_user_datasets(user_id: int):
    statement = select(Dataset).where(Dataset.owner_id == user_id)
    datasets = session.exec(statement).all()
    return datasets

# Удаление датасета
@dataset_router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Удаление файла с диска
    if os.path.exists(dataset.data_file_path):
        os.remove(dataset.data_file_path)

    session.delete(dataset)
    session.commit()
    return {"message": "Dataset deleted successfully"}

#скачивание датасета
@dataset_router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: int):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Проверяем, существует ли файл
    if not os.path.exists(dataset.data_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(dataset.data_file_path, media_type="application/octet-stream", filename=os.path.basename(dataset.data_file_path))