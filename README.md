# AI Surveillance System Tutorial with Hardware Compatibility and AI Suitability Insights

This tutorial provides a detailed, step-by-step guide to setting up an AI surveillance system using your existing hardware (Intel Core i9-10900K CPU, 64GB RAM, ASUS Prime Z590-A motherboard), the Reolink RLC-1240A camera, TP-Link PoE+ Injector, 1.82 TB hard drive, and Raspberry Pi Zero 2 WH. It includes code samples for key components, such as Python scripts for the Raspberry Pi and AWS integration. The focus is on cost-efficiency, using free/open-source tools like Shinobi for NVR and DeepStack for local AI processing, while integrating AWS for filtered event-based uploads.

## Hardware Compatibility Notes
Your ASUS Prime Z590-A motherboard is compatible with the recommended GPUs:

- **PCIe Slot Compatibility**: Primary PCIe 4.0 x16 slot runs at PCIe 3.0 with your i9-10900K. Both RTX 3060 and RTX 4060 are backward-compatible.
- **Power Supply Requirements**: Ensure PSU has 550W+ and an 8-pin PCIe connector.
- **Physical Fit**: Fits in most cases; check GPU dimensions.

If issues arise, update BIOS from ASUS website.

## Suitability of RTX 3060 and RTX 4060 for LLMs and AI Development
- **RTX 3060 (12GB VRAM)**: Good for small-to-medium LLMs (e.g., Llama 7B inference). Use Ollama for testing.
- **RTX 4060 (8GB VRAM)**: Suitable for beginners but VRAM-limited for larger models. Both support CUDA for PyTorch/TensorFlow.

## Step-by-Step Implementation Guide

### Phase 1: Hardware and NVR Software Setup

**Step 1: Install the GPU and Drivers**
1. Shut down PC, insert GPU into PCIe x16 slot.
2. Connect power, boot up.
3. Download drivers from NVIDIA site:  
   ```bash
   # Example for Windows: Run the installer.exe
   ```
4. Verify:  
   ```bash
   nvidia-smi  # In Command Prompt
   ```

**Step 2: Install Shinobi NVR Software**
1. Install Docker Desktop.
2. Run:  
   ```bash
   docker pull shinobicctv/shinobi
   docker run -d -p 8080:8080 -v /path/to/storage:/config --name shinobi shinobicctv/shinobi
   ```
3. Access http://localhost:8080, set up admin.

**Step 3: Configure the Reolink RLC-1240A Camera**
1. Connect via PoE injector.
2. Set static IP, enable RTSP in camera web interface.
3. RTSP URLs: Main - rtsp://<ip>:554/h265Preview_01_main; Sub - rtsp://<ip>:554/h264Preview_01_sub

**Step 4: Connect Shinobi to the Camera**
1. Add monitor in Shinobi UI, input RTSP URL.
2. Set recording to continuous, storage to 1.82 TB drive.

### Phase 2: Local AI Processing with DeepStack

**Step 5: Install and Configure DeepStack**
1. Install NVIDIA Container Toolkit.
2. Run:  
   ```bash
   docker pull deepquestai/deepstack:gpu
   docker run --gpus all -e VISION-DETECTION=True -p 5000:5000 deepquestai/deepstack:gpu
   ```
3. Test:  
   ```bash
   curl -X POST -F image=@test.jpg http://localhost:5000/v1/vision/detection
   ```

**Step 6: Integrate Shinobi and DeepStack**
Use Shinobi's plugin or a custom script. Example Python script to bridge:  
```python
import requests
import cv2

# Capture frame from Shinobi or directly
cap = cv2.VideoCapture('rtsp://...')
ret, frame = cap.read()
cv2.imwrite('frame.jpg', frame)

# Send to DeepStack
files = {'image': open('frame.jpg', 'rb')}
response = requests.post('http://localhost:5000/v1/vision/detection', files=files)
print(response.json())
if any(obj['label'] == 'person' for obj in response.json().get('predictions', [])):
    # Trigger alert
    pass
```

### Phase 3: Raspberry Pi Zero 2 WH as Edge Trigger

**Step 7: Set Up the Pi for Sub-Stream Monitoring**
1. Install Raspberry Pi OS.
2. Install deps:  
   ```bash
   sudo apt install python3-opencv ffmpeg
   ```
3. Script:  
```python
import cv2
import requests
import numpy as np
import time

RTSP_SUB = 'rtsp://<camera_ip>:554/h264Preview_01_sub'
DESKTOP_URL = 'http://<desktop_ip>:8080/trigger'  # Custom endpoint in Shinobi or Flask app

cap = cv2.VideoCapture(RTSP_SUB)
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(1)
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if prev_frame is None:
        prev_frame = gray
        continue
    
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    if np.sum(thresh) > 10000:  # Adjust threshold
        requests.post(DESKTOP_URL, json={'event': 'motion'})
    
    prev_frame = gray
    time.sleep(0.5)  # Poll every 0.5s
```

4. Run on boot: Add to crontab `@reboot python3 /path/to/script.py`

**Step 8: Pi as Trigger Mechanism**
- On desktop, set up a simple Flask server to receive trigger and activate DeepStack.  
```python
from flask import Flask, request
import subprocess  # To trigger Shinobi or DeepStack

app = Flask(__name__)

@app.route('/trigger', methods=['POST'])
def trigger():
    # Capture high-res frame and send to DeepStack
    # Example: Use ffmpeg to grab frame
    subprocess.run(['ffmpeg', '-i', 'rtsp://<main_stream>', '-vframes', '1', 'frame.jpg'])
    # Then send to DeepStack as above
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Phase 4: AWS Integration for Notifications

**Step 9: Set Up AWS Services**
1. Create S3 bucket, Lambda function, SNS topic.
2. Lambda example (Python):  
```python
import json
import boto3
from botocore.exceptions import ClientError

rekognition = boto3.client('rekognition')
sns = boto3.client('sns')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    try:
        response = rekognition.detect_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': key}},
            MaxLabels=10
        )
        if any(label['Name'] == 'Person' for label in response['Labels']):
            sns.publish(
                TopicArn='arn:aws:sns:your-region:account:topic',
                Message='Person detected!'
            )
    except ClientError as e:
        print(e)
    return {'statusCode': 200}
```

**Step 10: Desktop Script for AWS Upload**
```python
import boto3
import cv2

s3 = boto3.client('s3')

# After DeepStack detects person
cap = cv2.VideoCapture('rtsp://<main>')
ret, frame = cap.read()
cv2.imwrite('event.jpg', frame)  # Crop if needed

s3.upload_file('event.jpg', 'your-bucket', 'event.jpg')
```

**Step 11: Test the Full Workflow**
- Simulate motion, check logs, verify notifications.

## Summary and Tips
- Secure your setup with firewalls and VPN.
- For AI exploration, install Ollama: `curl https://ollama.ai/install.sh | sh` and run `ollama run llama2`.

