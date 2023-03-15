from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()
##change path to where you stored model
## the path should be "C:\\Users\\{Username}\\Documents\\GitHub\\agro_project\\models\\strawberry\\1"
MODEL = tf.keras.models.load_model("D:\\agro project\\Image Data base\\Image Data base\\models\\strawberry\\1")
CLASS_NAMES = ["Strawberry leaf scorch","Strawberry healthy"]
@app.get("/home")
async def root():
    return {"message": "Hello World"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    image = read_file_as_image(await file.read())
    img_batch=np.expand_dims(image, 0)
    predictions=MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)