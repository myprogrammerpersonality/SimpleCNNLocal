from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch
from PIL import Image
import cv2
import io
import base64
import numpy as np
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Set TORCH_HOME to the same directory as in the Dockerfile
# os.environ["TORCH_HOME"] = "/var/task/torch_cache"
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

COCO_LABELS = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def predict(img):
    tensor = torch.unsqueeze(torchvision.transforms.functional.to_tensor(img), 0)
    with torch.no_grad():
        prediction = model(tensor)
    return prediction

@app.get("/", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.post("/uploadfile/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print("content: ", contents)
        print("type: ", type(contents))
        nparr = np.frombuffer(contents, np.uint8)
        print("nparr: ", nparr)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("img.shape", img.shape)

        if img is None:
            raise HTTPException(status_code=400, detail="Unable to decode image")

        prediction = predict(img)

        print("^^", prediction)

        if not prediction:
            raise HTTPException(status_code=400, detail="Prediction failed")

        for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
            if score > 0.5:
                label_text = f"{COCO_LABELS[label.item()]}: {score.item():.2f}"
                print(label_text)
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(img, label_text, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        _, img_byte_arr = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(img_byte_arr.tobytes()).decode()
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        return templates.TemplateResponse("base.html", {"request": request, "image": img_data_url})

    except Exception as e:
        print(f"Error: {str(e)}") # print or log the error for debugging
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ping")
def read_root():
    return {"Hello": "World"}