import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from model import *
from json_database import JsonDatabase


def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model

def inference(image, threshold, file_name):
    model = load_trained_model('./models/FER_trained_model.pt')
    emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                    4: 'anger', 5: 'disgust', 6: 'fear'}

    print("Processing an image")
    print("Threshold: ", threshold)

    img_transform = transforms.Compose([
        transforms.ToTensor()])

    result = {
        'file_name': file_name,
        'threshold': threshold,
        'faces': []
    }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image)
    id_counter = 1
    for (x, y, w, h) in faces:
        face_instance = {}
        face_instance['face_id'] = id_counter
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        resize_image = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        X = resize_image/256
        X = Image.fromarray((X))
        X = img_transform(X).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            log_ps = model.cpu()(X)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            pred = emotion_dict[int(top_class.numpy())]

            for i in range(6):
                certainty = round(ps[0][i].item(), 2)
                if certainty >= threshold:
                    face_instance[emotion_dict[i]] = certainty

        face_instance['bbox'] = [str(x), str(y), str(x+w), str(y+h)]
        result['faces'].append(face_instance)
        id_counter += 1

        cv2.putText(image, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    Image.fromarray((image)).show()
    store_data_in_db(result)

def store_data_in_db(data):
    db_path = "images.db"
    with JsonDatabase("images", db_path) as db:
        db.add_item(data)
        db.print()