Below is a revised tutorial for setting up an AI surveillance system using your existing desktop configuration (Intel Core i9-10900K CPU, 64GB RAM, ASUS Prime Z590-A motherboard, 1.82 TB hard drive) without a dedicated GPU like the RTX 3060 or RTX 4060. The setup will rely on your CPU for all processing, including H.265 video decoding and DeepStack AI inference, and will use the same hardware (Reolink RLC-1240A camera, TP-Link PoE+ Injector, Raspberry Pi Zero 2 WH) and software (Shinobi, DeepStack, AWS). I’ll address whether this setup is viable with your CPU, provide detailed steps with code samples, and include a downloadable Markdown file by providing the content for you to copy and save. I’ll also revisit the suitability of your CPU for AI/LLM tasks without a GPU.

---

## AI Surveillance System Tutorial (CPU-Only) with Hardware Compatibility and AI Suitability Insights

This tutorial outlines how to set up an AI surveillance system using your desktop (Intel Core i9-10900K, 64GB RAM, ASUS Prime Z590-A, 1.82 TB HDD), Reolink RLC-1240A camera, TP-Link PoE+ Injector, and Raspberry Pi Zero 2 WH, without a dedicated GPU. It includes code samples for key components, such as Python scripts for the Raspberry Pi and AWS integration. The focus is on cost-efficiency with free/open-source tools (Shinobi for NVR, DeepStack for AI), integrating AWS for event-based notifications. The tutorial addresses the feasibility of running this system on your CPU and its suitability for AI/LLM development.

### Feasibility Without a GPU
Your Intel Core i9-10900K (10 cores, 20 threads, 3.7-5.3 GHz) and 64GB RAM are powerful enough to handle the surveillance system, but there are trade-offs:

- **H.265 Video Decoding**: The i9-10900K has Intel UHD Graphics 630 with Quick Sync Video, which supports hardware-accelerated H.265 decoding. This offloads video processing from the CPU cores, making it viable for decoding the 12MP stream from the Reolink RLC-1240A. However, Quick Sync is less efficient than NVIDIA’s NVDEC (on RTX GPUs), so CPU usage may spike during intensive tasks like simultaneous decoding and recording.

- **DeepStack AI Inference**: DeepStack’s CPU mode is 5-20x slower than GPU mode. Your i9-10900K can handle inference for 1-2 camera streams, but processing high-resolution frames for object detection (e.g., person detection) will be slower (0.5-2 seconds per frame vs. <0.1s with GPU). With a single camera, this is manageable, especially with the Raspberry Pi handling low-res motion triggers to reduce load.

- **Performance Limits**: Expect 20-40% CPU usage for Shinobi recording and decoding one 12MP stream, plus 10-20% per DeepStack inference (depending on frame rate). Your 10-core CPU and 64GB RAM can handle this for a single camera, but scaling to multiple cameras may strain resources, causing delays or dropped frames.

- **AI/LLM Suitability**: Without a GPU, LLM and AI tasks rely entirely on the CPU. The i9-10900K can run small LLMs (e.g., Llama 7B inference) using tools like Ollama or Hugging Face Transformers, leveraging its high core count and RAM. However, training or running larger models (e.g., Llama 13B+) will be slow, and some tasks (e.g., Stable Diffusion) are impractical without GPU acceleration. For hobbyist AI exploration, it’s sufficient but limited compared to GPU-based setups.

**Conclusion**: Your setup is viable for a single-camera surveillance system with AI, but inference speed and scalability are constrained. For AI/LLM work, the CPU is adequate for learning and small-scale inference but not ideal for heavy training or large models. If you plan to expand the system or dive deeper into AI, consider adding a GPU later.

### Prerequisites
- **Hardware**: Desktop (i9-10900K, 64GB RAM, Prime Z590-A, 1.82 TB HDD), Reolink RLC-1240A, TP-Link PoE+ Injector, Raspberry Pi Zero 2 WH.
- **Software**: Windows 11 (or Ubuntu 22.04), Docker Desktop, Python 3.9+, AWS CLI.
- **Network**: Stable LAN, camera on same subnet as desktop/Pi.

### Step-by-Step Implementation Guide
Total setup time: 3-5 hours. Assumes Windows 11; adapt for Linux if needed.

#### Phase 1: Hardware and NVR Software Setup

**Step 1: Verify CPU and System Setup**
1. Ensure your i9-10900K is running optimally:
   - Update BIOS to version 1201+ via ASUS website for stability.
   - In BIOS, enable Intel Quick Sync Video (ASUS UEFI → Advanced → System Agent → Graphics Configuration → iGPU Multi-Monitor → Enabled).
   - Check CPU performance: Run `taskmgr` (Windows) or `htop` (Linux) to confirm low idle usage.
2. Verify 1.82 TB HDD is accessible (e.g., `D:/Surveillance`).

**Step 2: Install Shinobi NVR Software**
1. Install Docker Desktop from docker.com (enable WSL2 if prompted).
2. Pull and run Shinobi:
   ```bash
   docker pull shinobicctv/shinobi
   docker run -d -p 8080:8080 -v D:/Surveillance:/config --name shinobi shinobicctv/shinobi
   ```
   - Maps `D:/Surveillance` to your HDD for recordings.
3. Access Shinobi at `http://localhost:8080`, create an admin account, and log in.
4. Configure storage: Set recording path to `/config/videos`.

**Step 3: Configure the Reolink RLC-1240A Camera**
1. Connect the camera to your router via the TP-Link PoE+ Injector:
   - Ethernet from router to injector’s “Data In.”
   - Ethernet from injector’s “PoE Out” to camera.
2. Power on; find the camera’s IP in your router’s DHCP list or Reolink app.
3. Access the camera’s web interface (`http://<camera_ip>`):
   - Set a static IP (e.g., 192.168.1.100).
   - Enable RTSP under Network → Advanced → RTSP.
   - Note URLs:
     - Main stream (12MP, H.265): `rtsp://<camera_ip>:554/h265Preview_01_main`
     - Sub-stream (720p, H.264): `rtsp://<camera_ip>:554/h264Preview_01_sub`
4. Test streams in VLC Media Player (`Media → Open Network Stream`).

**Step 4: Connect Shinobi to the Camera**
1. In Shinobi UI, go to “Monitors” → “Add Monitor.”
2. Enter:
   - Name: e.g., “FrontCamera”
   - Protocol: RTSP
   - Host: `<camera_ip>`
   - Port: 554
   - Path: `/h265Preview_01_main`
   - Stream Type: H.265
   - Recording: Continuous
3. Set storage to `/config/videos`.
4. Save and test live view. Verify recordings save to the HDD (30-day archive).
5. Monitor CPU usage in Task Manager; expect ~20-30% for one stream.

#### Phase 2: Local AI Processing with DeepStack (CPU Mode)

**Step 5: Install and Configure DeepStack**
1. Pull and run DeepStack CPU image (no GPU flags):
   ```bash
   docker pull deepquestai/deepstack:cpu
   docker run -e VISION-DETECTION=True -v D:/DeepStack:/datastore -p 5000:5000 deepquestai/deepstack:cpu
   ```
2. Test DeepStack:
   ```bash
   curl -X POST -F image=@test.jpg http://localhost:5000/v1/vision/detection
   ```
   - Upload a sample image; expect JSON with object labels (e.g., “person”). Note slower response (~0.5-2s).

**Step 6: Integrate Shinobi and DeepStack**
1. Create a Python script to bridge Shinobi and DeepStack:
   ```python
   import requests
   import cv2
   import time

   SHINOBI_RTSP = 'rtsp://<camera_ip>:554/h265Preview_01_main'
   DEEPSTACK_URL = 'http://localhost:5000/v1/vision/detection'

   def analyze_frame():
       cap = cv2.VideoCapture(SHINOBI_RTSP)
       ret, frame = cap.read()
       if not ret:
           cap.release()
           return False
       cv2.imwrite('frame.jpg', frame)
       cap.release()

       files = {'image': open('frame.jpg', 'rb')}
       response = requests.post(DEEPSTACK_URL, files=files)
       predictions = response.json().get('predictions', [])
       return any(obj['label'] == 'person' and obj['confidence'] > 0.8 for obj in predictions)

   while True:
       if analyze_frame():
           print("Person detected!")
           # Trigger alert (e.g., log or notify)
       time.sleep(2)  # Slower polling to reduce CPU load
   ```
2. Run: `python deepstack_bridge.py`.
3. Alternatively, use Shinobi’s webhook (Monitor Settings → Events → Webhook to `http://localhost:5000`). Adjust polling to balance CPU usage (e.g., 1-2 frames/sec).

#### Phase 3: Raspberry Pi Zero 2 WH as Edge Trigger

**Step 7: Set Up the Pi for Sub-Stream Monitoring**
1. Install Raspberry Pi OS Lite via Raspberry Pi Imager.
2. Connect to WiFi or Ethernet (use a USB Ethernet HAT if needed).
3. Install dependencies:
   ```bash
   sudo apt update && sudo apt install python3-opencv ffmpeg -y
   ```
4. Create a motion detection script:
   ```python
   import cv2
   import requests
   import numpy as np
   import time

   RTSP_SUB = 'rtsp://<camera_ip>:554/h264Preview_01_sub'
   DESKTOP_URL = 'http://<desktop_ip>:5001/trigger'

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
           try:
               requests.post(DESKTOP_URL, json={'event': 'motion'}, timeout=2)
               print("Motion detected, notified desktop")
           except Exception as e:
               print(f"Error: {e}")

       prev_frame = gray
       time.sleep(0.5)
   ```
5. Save as `motion.py`, run on boot:
   ```bash
   crontab -e
   # Add: @reboot python3 /home/pi/motion.py
   ```

**Step 8: Pi as Trigger Mechanism**
1. On the desktop, set up a Flask server:
   ```python
   from flask import Flask, request
   import subprocess
   import cv2
   import requests

   app = Flask(__name__)

   @app.route('/trigger', methods=['POST'])
   def trigger():
       # Capture high-res frame
       subprocess.run(['ffmpeg', '-i', 'rtsp://<camera_ip>:554/h265Preview_01_main', '-vframes', '1', 'frame.jpg'])
       # Send to DeepStack
       files = {'image': open('frame.jpg', 'rb')}
       response = requests.post('http://localhost:5000/v1/vision/detection', files=files)
       if any(obj['label'] == 'person' for obj in response.json().get('predictions', [])):
           print("Person detected, triggering AWS")
           # Call AWS upload (Step 10)
       return 'OK'

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5001)
   ```
2. Run: `python flask_server.py`.

#### Phase 4: AWS Integration for Notifications

**Step 9: Set Up AWS Services**
1. Sign up for AWS (free tier).
2. Create an S3 bucket (`surveillance-events`).
3. Create an SNS topic (`PersonDetection`); subscribe an email/SMS.
4. Create a Lambda function (Python 3.9):
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
                   TopicArn='arn:aws:sns:<region>:<account>:PersonDetection',
                   Message='Person detected at ' + key
               )
       except ClientError as e:
           print(e)
       return {'statusCode': 200}
   ```
5. Set S3 event trigger: Upload to bucket → invoke Lambda.

**Step 10: Desktop Script for AWS Upload**
1. Install AWS CLI and configure (`aws configure`).
2. Script to upload on DeepStack detection:
   ```python
   import boto3
   import cv2
   import time

   s3 = boto3.client('s3')

   def upload_to_aws():
       cap = cv2.VideoCapture('rtsp://<camera_ip>:554/h265Preview_01_main')
       ret, frame = cap.read()
       if ret:
           cv2.imwrite('event.jpg', frame)
           s3.upload_file('event.jpg', 'surveillance-events', 'event_' + str(int(time.time())) + '.jpg')
       cap.release()

   # Call from Step 8’s Flask trigger after DeepStack confirms person
   upload_to_aws()
   ```

**Step 11: Test the Full Workflow**
1. Simulate motion in front of the camera.
2. Check:
   - Pi logs (`cat /var/log/syslog | grep motion`).
   - Flask logs for trigger receipt.
   - DeepStack response (slower due to CPU).
   - AWS S3 for uploaded images.
   - SNS for notifications.
3. Verify Shinobi recordings (UI → Monitor → Playback).
4. Monitor CPU usage in Task Manager; adjust DeepStack polling if >50%.

### Summary and Tips
- **Workflow**: Camera → Shinobi (records) → Pi (triggers) → DeepStack (CPU-based AI) → AWS (notifications).
- **Costs**: Free locally; AWS ~$0.01-0.10/event.
- **Security**: Use VLAN for camera, HTTPS for Shinobi, AWS IAM roles.
- **AI/LLM**: Install Ollama for CPU-based LLM testing:
   ```bash
   curl https://ollama.ai/install.sh | sh
   ollama run mistral
   ```
- **Troubleshooting**: Check Docker logs (`docker logs shinobi`), ensure ports open (554, 8080, 5000, 5001), monitor CPU usage (`taskmgr`).

### Save as Markdown
Copy the content above and save it as `ai_surveillance_tutorial_cpu.md` in a text editor. If you need further customization or run into issues, let me know!
