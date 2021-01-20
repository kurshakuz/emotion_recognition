from __future__ import print_function
import requests
import json
import cv2
import numpy as np
import json     
import base64


threshold = 0.1
addr = 'http://localhost:5000'
main_url = addr + '/index'

file_path = './static/'
file_name = 'test.jpg'
# file_name = 'happy.jpeg'
# file_name = 'angry.jpeg'

image_file = file_path + file_name

with open(image_file, "rb") as f:
    im_bytes = f.read()        
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

payload = json.dumps({"image": im_b64, "threshold": threshold, "file_name": file_name})
response = requests.post(main_url, data=payload, headers=headers)

print(json.loads(response.text))