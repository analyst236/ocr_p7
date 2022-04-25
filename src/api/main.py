from fastapi import FastAPI
from xgboost import XGBClassifier
import pickle
import pandas as pd
from pydantic import BaseModel
from typing import Dict, List
import settings as conf


app = FastAPI()


class Data(BaseModel):
    data: Dict = None

class Chart(BaseModel):
    customer_id: int = None
    feature: List[str] = None

transformer = pickle.load(open(f"{conf.MODEL_PATH}/transformer_pipeline.pkl", 'rb'))
model = pickle.load(open(f"{conf.MODEL_PATH}/lgbm_undersample.pkl", 'rb'))
train_data = pd.read_csv(f"{conf.COMPUTE_DATA_PATH}/undersample_df_train_split.csv")
test_data = pd.read_csv(f"{conf.COMPUTE_DATA_PATH}/transform_df_test.csv")
raw_train = pd.read_csv(f"{conf.RAW_DATA_PATH}/application_train.csv")
raw_test = pd.read_csv(f"{conf.RAW_DATA_PATH}/application_test.csv")



@app.on_event("startup")
async def startup_event():
    pass
    print(train_data.shape)
    print(test_data.shape)
    print(model.predict_proba(test_data))


@app.get("/")
async def root():
    return {"message": "Hello  World"}


@app.get("/customers")
async def get_all_customers_id():
    result = test_data['SK_ID_CURR'].to_list()
    return result


@app.get("/detail/{customer_id}")
async def get_customer_detail_by_id(customer_id: int):
    raw = raw_test[raw_test['SK_ID_CURR'] == customer_id].fillna('NAN').to_dict('list')
    processed = test_data[test_data['SK_ID_CURR'] == customer_id].fillna('NAN').to_dict('list')

    return { 'raw' : raw, 'processed' : processed }


@app.get("/population")
async def get_population_data():
    return {'raw_population': raw_train.sample(10000).fillna('NAN').to_dict('list') }


@app.post("/predict")
async def predict_from_customer_detail(customer_details: Data):
    print(customer_details)

    customer_df = pd.DataFrame(customer_details.data)
    label = model.predict(customer_df).tolist()
    result = model.predict_proba(customer_df).tolist()
    read = lambda x: 'No Default' if x == 0 else 'Default'
    return { 'proba': result[0], 'label': read(label[0]) }


@app.post("/chart")
async def chart_customer(payload: Chart):
    result = {}
    df = raw_train.dropna(subset=payload.feature)
    result['population_0'] = df[payload.feature[0]].to_list()
    result['target'] = df['TARGET'].to_list()
    result['customer_0'] = raw_test[raw_test['SK_ID_CURR'] == payload.customer_id][payload.feature[0]].to_list()

    if len(payload.feature) > 1:
        result['population_1'] = df[payload.feature[1]].to_list()
        result['customer_1'] = raw_test[raw_test['SK_ID_CURR'] == payload.customer_id][payload.feature[1]].to_list()
    return result