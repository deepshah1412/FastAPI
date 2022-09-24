from typing import Optional
from fastapi import FastAPI
from fastapi.params import Body
from pydantic import BaseModel
from enum import Enum


app = FastAPI() #define fastAPI

class Post(BaseModel):
    title:str
    content:str
    published:bool = True     #By defaul True
    rating:Optional[int]=None #optional field

#request get method url:"/"

@app.get("/")   #path
async def root():  # function
    return {"message": "Hello API!"}

@app.get("/post")  #get method to read data.
def get_post():
    return{"data":"This is your post"} 

@app.post("/createposts") #post method to read data.
def create_posts(new_post:Post):
    print(new_post)
    print(new_post.dict()) #give us dict form
    return {"data":"new post"}

@app.get("/items/{item_id}")    
async def read_item(item_id:int):
    return {"item_id": item_id}
#title str,content str

@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

class ModelName(str, Enum):
    alexnet = "alexnet1"
    resnet = "resnet2"
    lenet = "lenet3"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


##Query Parameters






