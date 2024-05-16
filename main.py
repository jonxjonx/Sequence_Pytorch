import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

cap = cv2.VideoCapture(r"C:\Users\User\Desktop\test\rgb_5sec.mp4")
if not cap.isOpened():
    print("Error opening video file")

fps = int(cap.get(cv2.CAP_PROP_FPS))

# Counter for frame extraction
frame_count = 0

output_folder = r"C:\Users\User\Desktop\test\captured_images"
os.makedirs(output_folder, exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Save frame every second (adjust this as needed)
    if frame_count % fps == 0:
        frame_name = f"frame_{frame_count // fps +1}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_name), frame)
        print(f"Saved {frame_name}")

    frame_count += 1

cap.release()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp11/weights/best.pt')

sequence=[]

image_files = os.listdir(output_folder)
for i in image_files:
    img = os.path.join('captured_images', i)
    results = model(img)
    for detection in results.xyxy[0]:
        label = int(detection[5])  # Extract the label from the detection
        if label == 0:  
            sequence.append("Black")
        elif label == 1:  
            sequence.append("Blue")
        elif label == 2:  
            sequence.append("Green")
        elif label == 3:  
            sequence.append("Red")

if (sequence[3] == "Black" and sequence[4] == "Black"):    # C1, C2, C3, Black, Black
    print(sequence[0],",",sequence[1],",",sequence[2])

elif (sequence[2] == "Black" and sequence[3] == "Black"):  # C2, C3, Black, Black, C1
    print(sequence[4],",",sequence[0],",",sequence[1])

elif (sequence[1] == "Black" and sequence[2] == "Black"):  # C3, Black, Black, C1, C2
    print(sequence[3],",",sequence[4],",",sequence[0])

elif (sequence[0] == "Black" and sequence[1] == "Black"):  # Black, Black, C1, C2, C3
    print(sequence[2],",",sequence[3],",",sequence[4])

elif (sequence[0] == "Black" and sequence[4] == "Black"):  # Black, C1, C2, C3, Black
    print(sequence[1],",",sequence[2],",",sequence[3])
