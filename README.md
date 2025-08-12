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

To download this tutorial as a Markdown file, click here: <a href="data:text/markdown;base64,CiMgQUkgU3VydmVpbGxhbmNlIFN5c3RlbSBUdXRvcmlhbCB3aXRoIEhhcmR3YXJlIENvbXBhdGliaWxpdHkgYW5kIEFJIFN1aXRhYmlsaXR5IEluc2lnaHRzCgpUaGlzIHR1dG9yaWFsIHByb3ZpZGVzIGEgZGV0YWlsZWQsIHN0ZXAtYnktc3RlcCBndWlkZSB0byBzZXR0aW5nIHVwIGFuIEFJIHN1cnZlaWxsYW5jZSBzeXN0ZW0gdXNpbmcgeW91ciBleGlzdGluZyBoYXJkd2FyZSAoSW50ZWwgQ29yZSBpOS0xMDkwMEsgQ1BVLCA2NEdCIFJBTSwgQVNVUyBQcmltZSBaNTkwLUEgbW90aGVyYm9hcmQpLCB0aGUgUmVvbGluayBSTEMtMTI0MEEgY2FtZXJhLCBUUC1MaW5rIFBvRSsgSW5qZWN0b3IsIDEuODIgVEIgaGFyZCBkcml2ZSwgYW5kIFJhc3BiZXJyeSBQaSBaZXJvIDIgV0guIEl0IGluY2x1ZGVzIGNvZGUgc2FtcGxlcyBmb3Iga2V5IGNvbXBvbmVudHMsIHN1Y2ggYXMgUHl0aG9uIHNjcmlwdHMgZm9yIHRoZSBSYXNwYmVycnkgUGkgYW5kIEFXUyBpbnRlZ3JhdGlvbi4gVGhlIGZvY3VzIGlzIG9uIGNvc3QtZWZmaWNpZW5jeSwgdXNpbmcgZnJlZS9vcGVuLXNvdXJjZSB0b29scyBsaWtlIFNoaW5vYmkgZm9yIE5WUiBhbmQgRGVlcFN0YWNrIGZvciBsb2NhbCBBSSBwcm9jZXNzaW5nLCB3aGlsZSBpbnRlZ3JhdGluZyBBV1MgZm9yIGZpbHRlcmVkIGV2ZW50LWJhc2VkIHVwbG9hZHMuCgojIyBIYXJkd2FyZSBDb21wYXRpYmlsaXR5IE5vdGVzCllvdXIgQVNVUyBQcmltZSBaNTkwLUEgbW90aGVyYm9hcmQgaXMgY29tcGF0aWJsZSB3aXRoIHRoZSByZWNvbW1lbmRlZCBHUFVzOgoKLSAqKlBDSWUgU2xvdCBDb21wYXRpYmlsaXR5Kio6IFByaW1hcnkgUENJZSA0LjAgeDE2IHNsb3QgcnVucyBhdCBQQ0llIDMuMCB3aXRoIHlvdXIgaTktMTA5MDBLLiBCb3RoIFJUWCAzMDYwIGFuZCBSVFggNDA2MCBhcmUgYmFja3dhcmQtY29tcGF0aWJsZS4KLSAqKlBvd2VyIFN1cHBseSBSZXF1aXJlbWVudHMqKjogRW5zdXJlIFBTVSBoYXMgNTUwVysgYW5kIGFuIDgtcGluIFBDSWUgY29ubmVjdG9yLgotICoqUGh5c2ljYWwgRml0Kio6IEZpdHMgaW4gbW9zdCBjYXNlczsgY2hlY2sgR1BVIGRpbWVuc2lvbnMuCgpJZiBpc3N1ZXMgYXJpc2UsIHVwZGF0ZSBCSU9TIGZyb20gQVNVUyB3ZWJzaXRlLgoKIyMgU3VpdGFiaWxpdHkgb2YgUlRYIDMwNjAgYW5kIFJUWCA0MDYwIGZvciBMTE1zIGFuZCBBSSBEZXZlbG9wbWVudAotICoqUlRYIDMwNjAgKDEyR0IgVlJBTSkqKjogR29vZCBmb3Igc21hbGwtdG8tbWVkaXVtIExMTXMgKGUuZy4sIExsYW1hIDdCIGluZmVyZW5jZSkuIFVzZSBPbGxhbWEgZm9yIHRlc3RpbmcuCi0gKipSVFggNDA2MCAoOEdCIFZSQU0pKio6IFN1aXRhYmxlIGZvciBiZWdpbm5lcnMgYnV0IFZSQU0tbGltaXRlZCBmb3IgbGFyZ2VyIG1vZGVscy4gQm90aCBzdXBwb3J0IENVREEgZm9yIFB5VG9yY2gvVGVuc29yRmxvdy4KCiMjIFN0ZXAtYnktU3RlcCBJbXBsZW1lbnRhdGlvbiBHdWlkZQoKIyMjIFBoYXNlIDE6IEhhcmR3YXJlIGFuZCBOVlIgU29mdHdhcmUgU2V0dXAKCioqU3RlcCAxOiBJbnN0YWxsIHRoZSBHUFUgYW5kIERyaXZlcnMqKgoxLiBTaHV0IGRvd24gUEMsIGluc2VydCBHUFUgaW50byBQQ0llIHgxNiBzbG90LgoyLiBDb25uZWN0IHBvd2VyLCBib290IHVwLgozLiBEb3dubG9hZCBkcml2ZXJzIGZyb20gTlZJRElBIHNpdGU6ICAKICAgYGBgYmFzaAogICAjIEV4YW1wbGUgZm9yIFdpbmRvd3M6IFJ1biB0aGUgaW5zdGFsbGVyLmV4ZQogICBgYGAKNC4gVmVyaWZ5OiAgCiAgIGBgYGJhc2gKICAgbnZpZGlhLXNtaSAgIyBJbiBDb21tYW5kIFByb21wdAogICBgYGAKCioqU3RlcCAyOiBJbnN0YWxsIFNoaW5vYmkgTlZSIFNvZnR3YXJlKioKMS4gSW5zdGFsbCBEb2NrZXIgRGVza3RvcC4KMi4gUnVuOiAgCiAgIGBgYGJhc2gKICAgZG9ja2VyIHB1bGwgc2hpbm9iaWNjdHYvc2hpbm9iaQogICBkb2NrZXIgcnVuIC1kIC1wIDgwODA6ODA4MCAtdiAvcGF0aC90by9zdG9yYWdlOi9jb25maWcgLS1uYW1lIHNoaW5vYmkgc2hpbm9iaWNjdHYvc2hpbm9iaQogICBgYGAKMy4gQWNjZXNzIGh0dHA6Ly9sb2NhbGhvc3Q6ODA4MCwgc2V0IHVwIGFkbWluLgoKKipTdGVwIDM6IENvbmZpZ3VyZSB0aGUgUmVvbGluayBSTEMtMTI0MEEgQ2FtZXJhKioKMS4gQ29ubmVjdCB2aWEgUG9FIGluamVjdG9yLgoyLiBTZXQgc3RhdGljIElQLCBlbmFibGUgUlRTUCBpbiBjYW1lcmEgd2ViIGludGVyZmFjZS4KMy4gUlRTUCBVUkxzOiBNYWluIC0gcnRzcDovLzxpcD46NTU0L2gyNjVQcmV2aWV3XzAxX21haW47IFN1YiAtIHJ0c3A6Ly88aXA+OjU1NC9oMjY0UHJldmlld18wMV9zdWIKCioqU3RlcCA0OiBDb25uZWN0IFNoaW5vYmkgdG8gdGhlIENhbWVyYSoqCjEuIEFkZCBtb25pdG9yIGluIFNoaW5vYmkgVUksIGlucHV0IFJUU1AgVVJMLgoyLiBTZXQgcmVjb3JkaW5nIHRvIGNvbnRpbnVvdXMsIHN0b3JhZ2UgdG8gMS44MiBUQiBkcml2ZS4KCiMjIyBQaGFzZSAyOiBMb2NhbCBBSSBQcm9jZXNzaW5nIHdpdGggRGVlcFN0YWNrCgoqKlN0ZXAgNTogSW5zdGFsbCBhbmQgQ29uZmlndXJlIERlZXBTdGFjayoqCjEuIEluc3RhbGwgTlZJRElBIENvbnRhaW5lciBUb29sa2l0LgoyLiBSdW46ICAKICAgYGBgYmFzaAogICBkb2NrZXIgcHVsbCBkZWVwcXVlc3RhaS9kZWVwc3RhY2s6Z3B1CiAgIGRvY2tlciBydW4gLS1ncHVzIGFsbCAtZSBWSVNJT04tREVURUNUSU9OPVRydWUgLXAgNTAwMDo1MDAwIGRlZXBxdWVzdGFpL2RlZXBzdGFjazpncHUKICAgYGBgCjMuIFRlc3Q6ICAKICAgYGBgYmFzaAogICBjdXJsIC1YIFBPU1QgLUYgaW1hZ2U9QHRlc3QuanBnIGh0dHA6Ly9sb2NhbGhvc3Q6NTAwMC92MS92aXNpb24vZGV0ZWN0aW9uCiAgIGBgYAoKKipTdGVwIDY6IEludGVncmF0ZSBTaGlub2JpIGFuZCBEZWVwU3RhY2sqKgpVc2UgU2hpbm9iaSdzIHBsdWdpbiBvciBhIGN1c3RvbSBzY3JpcHQuIEV4YW1wbGUgUHl0aG9uIHNjcmlwdCB0byBicmlkZ2U6ICAKYGBgcHl0aG9uCmltcG9ydCByZXF1ZXN0cwppbXBvcnQgY3YyCgojIENhcHR1cmUgZnJhbWUgZnJvbSBTaGlub2JpIG9yIGRpcmVjdGx5CmNhcCA9IGN2Mi5WaWRlb0NhcHR1cmUoJ3J0c3A6Ly8uLi4nKQpyZXQsIGZyYW1lID0gY2FwLnJlYWQoKQpjdjIuaW13cml0ZSgnZnJhbWUuanBnJywgZnJhbWUpCgojIFNlbmQgdG8gRGVlcFN0YWNrCmZpbGVzID0geydpbWFnZSc6IG9wZW4oJ2ZyYW1lLmpwZycsICdyYicpfQpyZXNwb25zZSA9IHJlcXVlc3RzLnBvc3QoJ2h0dHA6Ly9sb2NhbGhvc3Q6NTAwMC92MS92aXNpb24vZGV0ZWN0aW9uJywgZmlsZXM9ZmlsZXMpCnByaW50KHJlc3BvbnNlLmpzb24oKSkKaWYgYW55KG9ialsnbGFiZWwnXSA9PSAncGVyc29uJyBmb3Igb2JqIGluIHJlc3BvbnNlLmpzb24oKS5nZXQoJ3ByZWRpY3Rpb25zJywgW10pKToKICAgICMgVHJpZ2dlciBhbGVydAogICAgcGFzcwpgYGAKCiMjIyBQaGFzZSAzOiBSYXNwYmVycnkgUGkgWmVybyAyIFdIIGFzIEVkZ2UgVHJpZ2dlcgoKKipTdGVwIDc6IFNldCBVcCB0aGUgUGkgZm9yIFN1Yi1TdHJlYW0gTW9uaXRvcmluZyoqCjEuIEluc3RhbGwgUmFzcGJlcnJ5IFBpIE9TLgoyLiBJbnN0YWxsIGRlcHM6ICAKICAgYGBgYmFzaAogICBzdWRvIGFwdCBpbnN0YWxsIHB5dGhvbjMtb3BlbmN2IGZmbXBlZwogICBgYGAKMy4gU2NyaXB0OiAgCmBgYHB5dGhvbgppbXBvcnQgY3YyCmltcG9ydCByZXF1ZXN0cwppbXBvcnQgbnVtcHkgYXMgbnAKaW1wb3J0IHRpbWUKClJUU1BfU1VCID0gJ3J0c3A6Ly88Y2FtZXJhX2lwPjo1NTQvaDI2NFByZXZpZXdfMDFfc3ViJwpERVNLVE9QX1VSTCA9ICdodHRwOi8vPGRlc2t0b3BfaXA+OjgwODAvdHJpZ2dlcicgICMgQ3VzdG9tIGVuZHBvaW50IGluIFNoaW5vYmkgb3IgRmxhc2sgYXBwCgpjYXAgPSBjdjIuVmlkZW9DYXB0dXJlKFJUU1BfU1VCKQpwcmV2X2ZyYW1lID0gTm9uZQoKd2hpbGUgVHJ1ZToKICAgIHJldCwgZnJhbWUgPSBjYXAucmVhZCgpCiAgICBpZiBub3QgcmV0OgogICAgICAgIHRpbWUuc2xlZXAoMSkKICAgICAgICBjb250aW51ZQogICAgCiAgICBncmF5ID0gY3YyLmN2dENvbG9yKGZyYW1lLCBjdjIuQ09MT1JfQkdSMkdSQVkpCiAgICBncmF5ID0gY3YyLkdhdXNzaWFuQmx1cihncmF5LCAoMjEsIDIxKSwgMCkKICAgIAogICAgaWYgcHJldl9mcmFtZSBpcyBOb25lOgogICAgICAgIHByZXZfZnJhbWUgPSBncmF5CiAgICAgICAgY29udGludWUKICAgIAogICAgZnJhbWVfZGVsdGEgPSBjdjIuYWJzZGlmZihwcmV2X2ZyYW1lLCBncmF5KQogICAgdGhyZXNoID0gY3YyLnRocmVzaG9sZChmcmFtZV9kZWx0YSwgMjUsIDI1NSwgY3YyLlRIUkVTSF9CSU5BUlkpWzFdCiAgICAKICAgIGlmIG5wLnN1bSh0aHJlc2gpID4gMTAwMDA6ICAjIEFkanVzdCB0aHJlc2hvbGQKICAgICAgICByZXF1ZXN0cy5wb3N0KERFU0tUT1BfVVJMLCBqc29uPXsnZXZlbnQnOiAnbW90aW9uJ30pCiAgICAKICAgIHByZXZfZnJhbWUgPSBncmF5CiAgICB0aW1lLnNsZWVwKDAuNSkgICMgUG9sbCBldmVyeSAwLjVzCmBgYAoKNC4gUnVuIG9uIGJvb3Q6IEFkZCB0byBjcm9udGFiIGBAcmVib290IHB5dGhvbjMgL3BhdGgvdG8vc2NyaXB0LnB5YAoKKipTdGVwIDg6IFBpIGFzIFRyaWdnZXIgTWVjaGFuaXNtKioKLSBPbiBkZXNrdG9wLCBzZXQgdXAgYSBzaW1wbGUgRmxhc2sgc2VydmVyIHRvIHJlY2VpdmUgdHJpZ2dlciBhbmQgYWN0aXZhdGUgRGVlcFN0YWNrLiAgCmBgYHB5dGhvbgpmcm9tIGZsYXNrIGltcG9ydCBGbGFzaywgcmVxdWVzdAppbXBvcnQgc3VicHJvY2VzcyAgIyBUbyB0cmlnZ2VyIFNoaW5vYmkgb3IgRGVlcFN0YWNrCgphcHAgPSBGbGFzayhfX25hbWVfXykKCkBhcHAucm91dGUoJy90cmlnZ2VyJywgbWV0aG9kcz1bJ1BPU1QnXSkKZGVmIHRyaWdnZXIoKToKICAgICMgQ2FwdHVyZSBoaWdoLXJlcyBmcmFtZSBhbmQgc2VuZCB0byBEZWVwU3RhY2sKICAgICMgRXhhbXBsZTogVXNlIGZmbXBlZyB0byBncmFiIGZyYW1lCiAgICBzdWJwcm9jZXNzLnJ1bihbJ2ZmbXBlZycsICctaScsICdydHNwOi8vPG1haW5fc3RyZWFtPicsICctdmZyYW1lcycsICcxJywgJ2ZyYW1lLmpwZyddKQogICAgIyBUaGVuIHNlbmQgdG8gRGVlcFN0YWNrIGFzIGFib3ZlCiAgICByZXR1cm4gJ09LJwoKaWYgX19uYW1lX18gPT0gJ19fbWFpbl9fJzoKICAgIGFwcC5ydW4oaG9zdD0nMC4wLjAuMCcsIHBvcnQ9ODA4MCkKYGBgCgojIyMgUGhhc2UgNDogQVdTIEludGVncmF0aW9uIGZvciBOb3RpZmljYXRpb25zCgoqKlN0ZXAgOTogU2V0IFVwIEFXUyBTZXJ2aWNlcyoqCjEuIENyZWF0ZSBTMyBidWNrZXQsIExhbWJkYSBmdW5jdGlvbiwgU05TIHRvcGljLgoyLiBMYW1iZGEgZXhhbXBsZSAoUHl0aG9uKTogIApgYGBweXRob24KaW1wb3J0IGpzb24KaW1wb3J0IGJvdG8zCmZyb20gYm90b2NvcmUuZXhjZXB0aW9ucyBpbXBvcnQgQ2xpZW50RXJyb3IKCnJla29nbml0aW9uID0gYm90bzMuY2xpZW50KCdyZWtvZ25pdGlvbicpCnNucyA9IGJvdG8zLmNsaWVudCgnc25zJykKCmRlZiBsYW1iZGFfaGFuZGxlcihldmVudCwgY29udGV4dCk6CiAgICBidWNrZXQgPSBldmVudFsnUmVjb3JkcyddWzBdWydzMyddWydidWNrZXQnXVsnbmFtZSddCiAgICBrZXkgPSBldmVudFsnUmVjb3JkcyddWzBdWydzMyddWydvYmplY3QnXVsna2V5J10KICAgIAogICAgdHJ5OgogICAgICAgIHJlc3BvbnNlID0gcmVrb2duaXRpb24uZGV0ZWN0X2xhYmVscygKICAgICAgICAgICAgSW1hZ2U9eydTM09iamVjdCc6IHsnQnVja2V0JzogYnVja2V0LCAnTmFtZSc6IGtleX19LAogICAgICAgICAgICBNYXhMYWJlbHM9MTAKICAgICAgICApCiAgICAgICAgaWYgYW55KGxhYmVsWydOYW1lJ10gPT0gJ1BlcnNvbicgZm9yIGxhYmVsIGluIHJlc3BvbnNlWydMYWJlbHMnXSk6CiAgICAgICAgICAgIHNucy5wdWJsaXNoKAogICAgICAgICAgICAgICAgVG9waWNBcm49J2Fybjphd3M6c25zOnlvdXItcmVnaW9uOmFjY291bnQ6dG9waWMnLAogICAgICAgICAgICAgICAgTWVzc2FnZT0nUGVyc29uIGRldGVjdGVkIScKICAgICAgICAgICAgKQogICAgZXhjZXB0IENsaWVudEVycm9yIGFzIGU6CiAgICAgICAgcHJpbnQoZSkKICAgIHJldHVybiB7J3N0YXR1c0NvZGUnOiAyMDB9CmBgYAoKKipTdGVwIDEwOiBEZXNrdG9wIFNjcmlwdCBmb3IgQVdTIFVwbG9hZCoqCmBgYHB5dGhvbgppbXBvcnQgYm90bzMKaW1wb3J0IGN2MgoKczMgPSBib3RvMy5jbGllbnQoJ3MzJykKCiMgQWZ0ZXIgRGVlcFN0YWNrIGRldGVjdHMgcGVyc29uCmNhcCA9IGN2Mi5WaWRlb0NhcHR1cmUoJ3J0c3A6Ly88bWFpbj4nKQpyZXQsIGZyYW1lID0gY2FwLnJlYWQoKQpjdjIuaW13cml0ZSgnZXZlbnQuanBnJywgZnJhbWUpICAjIENyb3AgaWYgbmVlZGVkCgpzMy51cGxvYWRfZmlsZSgnZXZlbnQuanBnJywgJ3lvdXItYnVja2V0JywgJ2V2ZW50LmpwZycpCmBgYAoKKipTdGVwIDExOiBUZXN0IHRoZSBGdWxsIFdvcmtmbG93KioKLSBTaW11bGF0ZSBtb3Rpb24sIGNoZWNrIGxvZ3MsIHZlcmlmeSBub3RpZmljYXRpb25zLgoKIyMgU3VtbWFyeSBhbmQgVGlwcwotIFNlY3VyZSB5b3VyIHNldHVwIHdpdGggZmlyZXdhbGxzIGFuZCBWUE4uCi0gRm9yIEFJIGV4cGxvcmF0aW9uLCBpbnN0YWxsIE9sbGFtYTogYGN1cmwgaHR0cHM6Ly9vbGxhbWEuYWkvaW5zdGFsbC5zaCB8IHNoYCBhbmQgcnVuIGBvbGxhbWEgcnVuIGxsYW1hMmAuCgo=" download="ai_surveillance_tutorial.md">Download Tutorial.md</a>
