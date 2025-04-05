from fastapi import APIRouter, HTTPException, Security, security, Depends
from fastapi.security import HTTPAuthorizationCredentials
from starlette.responses import JSONResponse
from starlette.status import HTTP_201_CREATED,HTTP_404_NOT_FOUND

from auth.auth import AuthHandler
from db.db import session
from models.user_models import UserInput, User, UserLogin
from repos.user_repository import select_all_users, find_user

user_router = APIRouter()
auth_handler = AuthHandler()


@user_router.post('/registration', status_code=201, tags=['Users'],
                  description='Register new user')
async def register(user: UserInput):
    users = select_all_users()
    if any(x.name == user.name for x in users):
        raise HTTPException(status_code=400, detail='Username is taken')
    hashed_pwd = auth_handler.get_password_hash(user.password)
    u = User(name=user.name, password=hashed_pwd)
    session.add(u)
    session.commit()
    return JSONResponse(status_code=HTTP_201_CREATED, content={"detail": "User created"})


@user_router.post('/login', tags=['Users'])
async def login(user: UserLogin):
    user_found = find_user(user.name)
    if not user_found:
        raise HTTPException(status_code=401, detail='Invalid username and/or password')
    verified = auth_handler.verify_password(user.password, user_found.password)
    if not verified:
        raise HTTPException(status_code=401, detail='Invalid username and/or password')
    token = auth_handler.encode_token(user_found.name)
    return {'token': token}


@user_router.get('/users/me', tags=['Users'])
async def get_current_user(user: User = Depends(auth_handler.get_current_user)):
    return user