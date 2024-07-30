import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import base64

# Load the cascade files for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to select the detector type
def select_detector():
    choice = st.sidebar.selectbox("Select detector type", ("Face", "Eyes"))
    return 1 if choice == "Face" else 2

# Function to create directory if it doesn't exist
def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Function to set the background image
def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    background-position: center;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to resize the image to fit the background dimensions
def resize_image(image_file, target_width, target_height):
    image = Image.open(image_file)
    image = image.resize((target_width, target_height), Image.LANCZOS)
    resized_image_path = "resized_background.png"
    image.save(resized_image_path)
    return resized_image_path

# Function to check for available cameras
def is_camera_available(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return False
    cap.release()
    return True

# Select the detector type
detector_type = select_detector()

# Create a directory for saving images
output_dir = st.sidebar.text_input("Enter directory name to save images", "output_images")
create_directory(output_dir)

# Background image upload
background_img = st.sidebar.file_uploader("Upload a background image", type=['png', 'jpg', 'jpeg'])

if background_img:
    # Resize the background image to 500x500 pixels
    resized_image_path = resize_image(background_img, 800, 900)
    set_background(resized_image_path)

# Check if camera is available
if not is_camera_available(0):
    st.error("No available camera found at index 0")
else:
    # Open the camera
    cap = cv2.VideoCapture(0)

    st.title("Real-Time Face or Eyes Detection and blur")

    run = st.checkbox('Run')
    save_image = st.button('Save Image')
    image_name = st.text_input("Enter image name", "")

    FRAME_WINDOW = st.image([])

    try:
        while run:
            # Capture frame-by-frame
            ret, img = cap.read()

            if not ret:
                st.error("Failed to capture image")
                break

            # Choose the cascade classifier based on user's choice
            if detector_type == 1:
                cascade = face_cascade
            else:
                cascade = eyes_cascade

            # Detect faces or eyes
            detections = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4)

            if detector_type == 1:
                # Face detection: add a box around the face
                for (x, y, w, h) in detections:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    roi = img[y:y + h, x:x + w]
                    blur = cv2.GaussianBlur(roi, (91, 91), 0)
                    img[y:y + h, x:x + w] = blur
                if len(detections) == 0:
                    cv2.putText(img, 'No Face Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

            else:
                # Eye detection: blur the eyes and add a box around them
                for (x, y, w, h) in detections:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    roi = img[y:y + h, x:x + w]
                    blur = cv2.GaussianBlur(roi, (25, 25), 0)
                    img[y:y + h, x:x + w] = blur
                if len(detections) == 0:
                    cv2.putText(img, 'No Eyes Found!', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

            # Convert the image to RGB (OpenCV uses BGR by default)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Display the resulting frame
            FRAME_WINDOW.image(img_pil)

            if save_image:
                if image_name:
                    image_path = os.path.join(output_dir, f"{image_name}.png")
                    if not os.path.exists(image_path):  # Check if the image already exists
                        img_pil.save(image_path)
                        st.success(f'Image saved successfully as {image_name}.png')
                    else:
                        st.warning(f'Image with name {image_name}.png already exists. Please use a different name.')
                else:
                    st.warning('Please enter an image name to save.')
                break  # Exit the loop after saving the image
    finally:
        # Release the camera
        cap.release()
