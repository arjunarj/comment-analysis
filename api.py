from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from model import predcit as ml
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Comment(BaseModel):
    comment: str

@app.get("/")
def root():
    return "👍"

@app.post("/predict")
def predicts(val: Comment):
    value = ml(val.comment)
    print(value)
    return value