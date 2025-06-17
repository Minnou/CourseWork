from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime
from sqlmodel import select
from models.dataset_models import Dataset
from models.prediction_models import Prediction
from models.ml_models import MLModel
from models.user_models import User
from db.db import session
from auth.auth import AuthHandler
from repos.prediction_repository import predict, read_prediction
import os

prediction_router = APIRouter(prefix="/predictions", tags=["Predictions"])
auth_handler = AuthHandler()

@prediction_router.post("/predict/{model_id}/{days}")
async def make_prediction(model_id: int, days: int, user: User = Depends(auth_handler.get_current_user)):
    model: MLModel = session.get(MLModel, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model or Dataset not found")
    
    dataset: Dataset = session.get(Dataset, model.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Model or Dataset not found")
    
    if dataset.owner_id != user.id or model.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Model or Dataset not found")

    result_path = predict(model.model_path, dataset.file_path, days)

    prediction = Prediction(model_id=model.id, result_file_path=result_path, owner_id=user.id, 
                            input_dataset_id=dataset.id, created_at=datetime.utcnow())
    session.add(prediction)
    session.commit()
    session.refresh(prediction)

    return prediction

@prediction_router.get("/{prediction_id}/download")
async def download_prediction(prediction_id: int, user: User = Depends(auth_handler.get_current_user)):
    prediction: Prediction = session.get(Prediction, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if prediction.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return FileResponse(prediction.result_file_path, media_type="application/octet-stream", filename=os.path.basename(prediction.result_file_path))

@prediction_router.get("/user")
async def get_user_predictions( user: User = Depends(auth_handler.get_current_user)):
    predictions = session.exec(select(Prediction).where(Prediction.owner_id == user.id)).all()
    return predictions

@prediction_router.delete("/delete/{prediction_id}")
async def delete_prediction(prediction_id: int, user: User = Depends(auth_handler.get_current_user)):
    prediction: Prediction =  session.get(Prediction, prediction_id)
    if not prediction or prediction.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if os.path.exists(prediction.result_file_path):
        os.remove(prediction.result_file_path)

    session.delete(prediction)
    session.commit()
    return {"detail": "Prediction deleted"}

@prediction_router.get("/{prediction_id}/data")
async def get_dataset_data(prediction_id: int, user: User = Depends(auth_handler.get_current_user)):
    prediction = session.exec(select(Prediction).where(Prediction.id == prediction_id)).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if (prediction.owner_id != user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(prediction.result_file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")

    return read_prediction(prediction.result_file_path)