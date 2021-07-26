from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

py_beer = "../models/pytorch_beer_v2.pt"

@app.get("/")
def read_root():
    return  {"""Beer style prediction : predict a type of beer based on some users’ rating criteria such as appearance, aroma, palate or taste.\n
             Github link: https://github.com/pri-mdsi/beerprediction
             Expected input parameters: \n
             Brewery id: [brewery_id] \n
             Review of Aroma: [review_aroma]\n
             Review of Appearance: [review_appearance]\n
             Review of Palate: [review_palate] \n
             Review of Taste: [review_taste] \n
             Alcohol by volume measure: [beer_abv] \n
 
            \n
             Go to /beer/type for prediction for a single input only\n
             Go to /beers/type for predictions for a multiple inputs\n
            """}
    

  
@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer style prediction is ready to go!!'


def format_features(review_aroma: int, review_appearance: int, review_palate: int, review_taste: int,beer_abv: int,brewery_id_new_enc: int):
  return {
        'Brewery id': [brewery_id_new_enc],
        'Review of Aroma': [review_aroma],
        'Review of Appearance': [review_appearance],
        'Review of Palate': [review_palate], 
        'Review of Taste' : [review_taste],
        'Alcohol by volume measure': [beer_abv]
}

@app.get("beer/type/")
def predict(review_aroma: int, review_appearance: int, review_palate: int, review_taste: int,beer_abv: int,brewery_id_new_enc: int):
    features = format_features(review_aroma,review_appearance, review_palate, review_taste,beer_abv,brewery_id_new_enc)
    obs = pd.DataFrame(features)
    pred = py_beer.predict(obs)
    return JSONResponse(pred.tolist())


    