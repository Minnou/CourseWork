from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import select
from models.dataset_models import Dataset
from models.ml_models import MLModel
from models.user_models import User
from db.db import session
from auth.auth import AuthHandler
import os
import time
from repos.ml_repository import train_arima_with_params, train_svr_with_params, find_best_arima_params, find_best_svr_params


ml_router = APIRouter(prefix="/ml", tags=["ML"])
auth_handler = AuthHandler()

MODEL_DIR = "./user_data/saved_models"

@ml_router.get("/find-best-params-svr/{dataset_id}")
async def find_best_params_svr(dataset_id: int, user: User = Depends(auth_handler.get_current_user)):
    dataset = session.exec(select(Dataset).where(Dataset.id == dataset_id)).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if (dataset.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return find_best_svr_params(dataset.file_path)

@ml_router.get("/find-best-params-arima/{dataset_id}")
async def find_best_params_arima(dataset_id: int, user: User = Depends(auth_handler.get_current_user)):
    dataset = session.exec(select(Dataset).where(Dataset.id == dataset_id)).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if (dataset.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return find_best_arima_params(dataset.file_path)


@ml_router.post("/train-arima/{dataset_id}")
async def train_with_params_arima(dataset_id: int, order: str = None, user: User = Depends(auth_handler.get_current_user)):
    dataset = session.exec(select(Dataset).where(Dataset.id == dataset_id)).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if (dataset.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    save_path = os.path.join(MODEL_DIR, f"ARIMA_{user.name}_{int(time.time())}.pkl")

    if not order:
        raise HTTPException(status_code=400, detail="ARIMA requires order and seasonal_order")
    
    order_tuple = tuple(map(int, order.split('_')))
    train_arima_with_params(dataset.file_path, save_path, order_tuple)

    ml_model = MLModel(
        name=f"ARIMA_{user.name}_{int(time.time())}",
        dataset_id=dataset.id,
        owner_id=user.id,
        model_path=save_path
    )
    session.add(ml_model)
    session.commit()
    session.refresh(ml_model)

    return {"message": "Model trained and saved", "model_id": ml_model.id}


@ml_router.post("/train-svr/{dataset_id}")
async def train_with_params_svr(dataset_id: int, kernel: str = None, C: float = None, epsilon: float = None,user: User = Depends(auth_handler.get_current_user)):
    dataset = session.exec(select(Dataset).where(Dataset.id == dataset_id)).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if (dataset.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    save_path = os.path.join(MODEL_DIR, f"SVR_{user.name}_{int(time.time())}.pkl")

    if not (kernel and C and epsilon):
        raise HTTPException(status_code=400, detail="SVR requires kernel, C, and epsilon")
    train_svr_with_params(dataset.file_path, save_path, kernel, C, epsilon)

    ml_model = MLModel(
        name=f"SVR_{user.name}_{int(time.time())}",
        dataset_id=dataset.id,
        owner_id=user.id,
        model_path=save_path
    )
    session.add(ml_model)
    session.commit()
    session.refresh(ml_model)

    return {"message": "Model trained and saved", "model_id": ml_model.id}


@ml_router.delete("/delete/{model_id}")
def delete_ml_model(model_id: int, user: User = Depends(auth_handler.get_current_user)):
    model = session.exec(select(MLModel).where(MLModel.id == model_id)).first()
    if not model or model.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Model not found")

    if os.path.exists(model.model_path):
        os.remove(model.model_path)

    session.delete(model)
    session.commit()
    return {"message": "ML model deleted successfully"}

@ml_router.get("/user")
def get_user_models(user: User = Depends(auth_handler.get_current_user)):
    models = session.exec(select(MLModel).where(MLModel.owner_id == user.id)).all()
    return models

@ml_router.get("/download/{model_id}")
def download_model(model_id: int, user: User = Depends(auth_handler.get_current_user)):
    model = session.exec(select(MLModel).where(MLModel.id == model_id)).first()
    
    if not model or model.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(model.model_path, filename=os.path.basename(model.model_path), media_type="application/octet-stream")