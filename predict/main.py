from fastapi import FastAPI, Depends
from starlette.requests import Request
from authenticator import check_request_origin


app = FastAPI()


@app.get("/")
def default():
    return {"model_server_active": True}


@app.get("/predict")
async def predict(request: Request = Depends(check_request_origin)):
    return {"prediction": "some prediction", "request_host": request.client.host}
