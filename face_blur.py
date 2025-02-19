# face_blur.py
import cv2
import torch
from ultralytics import YOLO
from shared import processing_progress

def initialize_model():
    model = YOLO('yolov8n-face.pt')
    if torch.cuda.is_available():
        model.to('cuda')
    return model

# def blur_face(image, box):
#     (startX, startY, endX, endY) = box.astype("int")
#     if startY >= endY or startX >= endX:
#         return image

#     face = image[startY:endY, startX:endX]
#     face = cv2.GaussianBlur(face, (99, 99), 30)
#     image[startY:endY, startX:endX] = face
#     return image


# Function to apply blur to detected faces
def blur_face(image, box):
    

    (startX, startY, endX, endY) = box.astype("int")
    if startY >= endY or startX >= endX:
        return image

    # Extract the face region
    face = image[startY:endY, startX:endX]
    
    # Create an empty mask for the face region
    mask = image.copy()
    mask[:] = 0
    
    # Calculate the center and size of the ellipse (approximating face shape)
    center = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)
    axes = ((endX - startX) // 2, (endY - startY) // 2)
    
    # Draw an ellipse on the mask, covering the face region
    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-1)
    
    # Apply GaussianBlur to the face region
    blurred_face = cv2.GaussianBlur(face, (69, 69), 30)
    
    # Extract the corresponding part of the mask (same size as face)
    mask_face_region = mask[startY:endY, startX:endX]
    
    # Combine the blurred face with the original image using the mask
    image[startY:endY, startX:endX] = cv2.bitwise_and(blurred_face, mask_face_region) + cv2.bitwise_and(face, cv2.bitwise_not(mask_face_region))
    
    return image

def process_video(input_path, output_path, model, filename):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        temp = results.numpy()

        print(temp.shape)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            print(boxes.shape)
            for box in boxes:
                print(box.shape)
                
                frame = blur_face(frame, box)

        out.write(frame)
        frame_count += 1
        progress = int((frame_count / total_frames) * 100)
        processing_progress[filename] = progress

    cap.release()
    out.release()
    processing_progress[filename] = 100
    print(f"Video processing complete. Output saved to {output_path}")