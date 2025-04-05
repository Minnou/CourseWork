from sqlmodel import create_engine
from secret import db_path #Путь до БД. СЕКРЕТНЫЙ ФАЙЛ
from sqlmodel import Session

engine = create_engine(f"sqlite:///{db_path}", echo=True)
session = Session(bind=engine)