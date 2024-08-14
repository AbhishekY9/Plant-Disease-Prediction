import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi import Request
import base64

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/fig", StaticFiles(directory="fig"), name="fig")

MODEL = tf.keras.models.load_model("../saved_models/1/model.keras")    #path where model is saved
CLASS_NAME = ["Early blight", "Late blight", "Healthy"]

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    # Read the image data for prediction
    image_data = await file.read()
    image = read_file_as_image(image_data)
    img_batch = np.expand_dims(image, 0)

    # Make the prediction
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAME[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    # Convert the image to base64 to pass to the template
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_base64": image_base64,
        "predicted_class": predicted_class,
        "confidence": str(round(float(confidence)*100,2))+"%"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)



