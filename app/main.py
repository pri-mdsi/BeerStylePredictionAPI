from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

beer_py = load('../models/pytorch_beer_9.5.pt')


@app.get("/")
def read_root():
    return {"Hello": "World"}


  
@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer style prediction is ready to go!'

    

