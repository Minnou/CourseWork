from fastapi import FastAPI
import uvicorn
from endpoints.dataset_endpoints import dataset_router
from endpoints.user_endpoints import user_router
from endpoints.ml_endpoits import ml_router
from endpoints.prediction_endpoints import prediction_router
#from db.db import engine
#from sqlmodel import SQLModel
#from models.dataset_models import *
#from models.ml_models import *
#from models.prediction_models import *
#from models.user_models import *

app = FastAPI()
#session = Session(bind=engine)
app.include_router(dataset_router)
app.include_router(user_router)
app.include_router(ml_router)
app.include_router(prediction_router)

#def create_db_and_tables():
#   SQLModel.metadata.create_all(engine)



if __name__ == "__main__":
#   create_db_and_tables()
   uvicorn.run("main:app", host="localhost", port=8000, reload=True)
