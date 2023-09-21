# Importing necessary library to complete this project
import cv2  # importing this for the computer vision task
import os  # importing this for the operating system function
import tkinter as tk  # this is to implement the GUI
from tkinter import filedialog  # this is to use in GUI for selecting videos
from tkinter import ttk  # this is thmed tkinter widgets
from PIL import Image, ImageTk  # this is for the image processing

# Loading the object and motion detection model which are in same directory
# thi is an configuration file for the model
mobil_net_config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# this is to load pre-trained frozen inference graph
frozen_model = "frozen_inference_graph.pb"

# here creating/making a instance of object detection model using Opencv's dnn module
detect_model = cv2.dnn_DetectionModel(frozen_model, mobil_net_config_file)
# Initializing an empty list classLabels.
labels_object = []
# Assigning the file "object.txt" to the variable.
file_name = "object.txt"
# Opening the file 'file_name' in read ('r') mode as a text ('t') file,
#  then assigning  file object to the variable 'fpt'.
# The 'with' statement is to  ensure that the file is properly closed after its suite finishes executing.
with open(file_name, 'rt') as fpt:
    # Reading the contents of the file 'fpt', removing trailing newline characters ('\n') at the end,
    # and spliting the remaining string into a list of strings using newline characters ('\n') as the delimiter.
    # Assigning the resulting list to classLabels.
    labels_object = fpt.read().rstrip('\n').split('\n')

# Setting the input size for the 'model' to a width of 320 pixels and a height of 320 pixels.
detect_model.setInputSize(320, 320)
# Setting the input scale for the 'model' to 1.0 divided by 127.5.
detect_model.setInputScale(1.0 / 127.5)
# Setting the input mean for the 'model' to the RGB values (127.5, 127.5, 127.5).
detect_model.setInputMean((127.5, 127.5, 127.5))
# Enabling swapping of color channels in the input for the 'model'.
detect_model.setInputSwapRB(True)


# Create a function to perform object and detection
def perform_object_detection(video_source=None):
    # Open the video file using the selected path from the GUI
    video_path = None  # Initialize video_path as None
    # Checking if the 'video_source' variable is not None before proceeding.
    if video_source is not None:
        # Code to be executed if 'video_source' is not None.
        # Assigning the value of the 'video_source' variable to 'video_path'.
        video_path = video_source
    else:
        # If 'video_source' is None, prompt the user to select a video file using a file dialog.
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        # Checking if the 'video_path' variable is empty or evaluates to False
        if not video_path:
            # Code executes if 'video_path' is empty or False.
            return

    # Createing a VideoCapture object 'cap' to open and read the video file 'video_path'.
    cap = cv2.VideoCapture(video_path)
    # Checking if the 'cap' VideoCapture object is not opened (failed to open the video file).
    if not cap.isOpened():  # Code to be executed if the video file failed to open
        # Creating a VideoCapture object 'cap' to open and read the video file 'vvv.mp4'.
        cap = cv2.VideoCapture('vvv.mp4')
    # Checking if the 'cap' VideoCapture object is not opened (failed to open the video file).
    if not cap.isOpened():  # Code executes if the video file failed to open.
        # Raising an IOError exception with the message "Cannot open video."
        raise IOError("Cannot open video")

    # Creating the output folder for captured images
    output_folder = "captured_objects"
    # Creating the directory 'output_folder' if it doesn't exist.
    # If 'output_folder' already exists, do nothing due to the 'exist_ok=True' parameter.
    os.makedirs(output_folder, exist_ok=True)

    # Define the desired width and height for resizing
    desired_width = 1000
    desired_height = 600

    font_scale = 3
    # Set the font for text rendering to FONT_HERSHEY_PLAIN.
    font = cv2.FONT_HERSHEY_PLAIN

    # Creating a canvas which will be used to display the video
    canvas = tk.Canvas(root, width=desired_width, height=desired_height)
    # Packing the 'canvas' widget to make it visible within its parent container.
    canvas.pack()

    while True:
        # Reading a frame from 'cap'. 'ret' indicates whether the read operation was successful,
        # and 'frame' contains the captured frame.
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame to the desired width and height
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Get the left half of the frame
        left_half_frame = frame[:, :desired_width // 2]

        # Detect objects in the left half of the frame
        ClassIndex, confidence, bbox = detect_model.detect(frame, confThreshold=0.55)

        # Checking if the length of the 'ClassIndex' list is not equal to zero (i.e., it contains elements).
        if len(ClassIndex) != 0:
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                # Checking if the value of 'ClassInd' is less than or equal to 80 (assuming it represents a class index).
                if ClassInd <= 80:
                    # Retrieving the corresponding object label from the 'classLabels' list.
                    object_label = labels_object[ClassInd - 1]
                    # Drawing a rectangle around the detected object using 'cv2.rectangle'.
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    # Displaying the object label as text near the object using 'cv2.putText'.
                    cv2.putText(frame, object_label, (boxes[0] + 10, boxes[1] + 48), font, fontScale=font_scale,
                                color=(0, 255, 0), thickness=3)

                    # Checking if the left edge of the bounding box is on the left side of the desired width divided by 2.
                    if boxes[0] < desired_width // 2:
                        # Extracting the region of interest (ROI) corresponding to the detected person.
                        person_image = frame[boxes[1]:boxes[3], boxes[0]:boxes[2]]
                        # Applying brightness, saturation, and contrast adjustments to the person image.
                        brightness_factor = 1.5
                        saturation_factor = 1.2
                        contrast_factor = 1.3
                        # Applying brightness adjustment using 'cv2.convertScaleAbs'
                        person_image = cv2.convertScaleAbs(person_image, alpha=brightness_factor, beta=0)
                        # Applying saturation adjustment using 'cv2.convertScaleAbs'
                        person_image = cv2.convertScaleAbs(person_image, alpha=saturation_factor, beta=0)
                        # Applying contrast adjustment using 'cv2.convertScaleAbs'
                        person_image = cv2.convertScaleAbs(person_image, alpha=contrast_factor, beta=0)

                        # Checkig if 'person_image' is not None and its size is not zero (it contains image data).
                        if not person_image is None and person_image.size != 0:
                            # Generating a unique image name based on the number of existing files in the 'output_folder'
                            image_name = f"person_{len(os.listdir(output_folder)) + 1}.jpg"
                            # Saving the 'person_image' to a JPEG file in the 'output_folder'.
                            cv2.imwrite(os.path.join(output_folder, image_name), person_image)
                            # Displaying an image from the 'output_folder' based on the given image index.
                            display_image(len(os.listdir(output_folder)) + 1)
        # Converting the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Creating a PIL  image from the RGB frame
        frame_pil = Image.fromarray(frame_rgb)
        # Creating a Tkinter-compatible PhotoImage from the PIL image.
        frame_tk = ImageTk.PhotoImage(image=frame_pil)

        # Display the video frame on the canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
        # Updating the 'canvas' to display the newly added image.
        canvas.update()

        # Waiting for a key press using 'cv2.waitKey(2)' and checking if the pressed key is 'q' (ASCII code 113).
        # If 'q' is pressed, exit the loop.
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

            # Releasing the video capture 'cap' to free up system resources.
    cap.release()
    # Closing all OpenCV windows and destroying the associated GUI elements.
    cv2.destroyAllWindows()


# Creating the main GUI window.
root = tk.Tk()
root.title("Object Detection GUI")


# Defining a function 'open_camera' to start object detection using the default camera (usually the built-in webcam).
def open_camera():
    perform_object_detection(0)  # Pass 0 to use the default camera (usually the built-in webcam)


# Creating a Tkinter button widget named 'open_camera_button' with the text "Open Camera."
# When the button is clicked, it calls the 'perform_object_detection' function with 0 as an argument.
# to use the default camera (usually the built-in webcam).
open_video_button = tk.Button(root, text="Open Video File", command=perform_object_detection)
open_video_button.pack()

# Creating a Tkinter button widget named 'open_camera_button' with the text "Open Camera."
# When the button is clicked, it calls the 'perform_object_detection' function with 0 as an argument
# to use the default camera (usually the built-in webcam).
open_camera_button = tk.Button(root, text="Open Camera", command=lambda: perform_object_detection(0))
open_camera_button.pack()

# Creating a Tkinter LabelFrame named 'image_frame' with the text "Captured Images."
# This frame is used to display captured images.
image_frame = ttk.LabelFrame(root, text="Captured Images:")
image_frame.pack(padx=10, pady=10)

# Defining the directory where captured object images are stored.
captured_objects_dir = "captured_objects"
# Listing the image files in the 'captured_objects' directory.
captured_image_files = os.listdir(captured_objects_dir)
# Initializing the current image index to 0.
current_image_index = 0


# Defining a function 'display_image' to display an image based on the given 'index'.
# The 'index' parameter specifies the position of the image in the list of captured images.
# Declaring the 'current_image_index' as a global variable within the function.
def display_image(index):
    global current_image_index
    # Checking if the 'index' is within the valid range of captured image files.
    if 0 <= index < len(captured_image_files):
        # Setting the 'current_image_index' to the specified 'index'.
        current_image_index = index
        # Getting the filename of the image at the specified 'index'.
        image_file = captured_image_files[index]
        # Constructing the full path to the image file.
        image_path = os.path.join(captured_objects_dir, image_file)
        # Opening the image using PIL .
        image = Image.open(image_path)
        # Creating a thumbnail of the image with a maximum size of (200, 200).
        image.thumbnail((200, 200))
        # Converting the PIL image to a Tkinter-compatible PhotoImage.
        photo = ImageTk.PhotoImage(image)

        # Creating a Tkinter label widget named 'label' within the 'image_frame' and displaying the 'photo' image.
        # The 'photo' attribute is used to keep a reference to the PhotoImage to prevent it from being garbage
        # collected. The label is placed in the grid with the 'row' and 'column'.
        label = tk.Label(image_frame, image=photo)
        # Storing a reference to the 'photo' to prevent it from being garbage collected.
        label.photo = photo
        # Placeing the label in the grid within the 'image_frame' at the specified 'row' and 'column' (index).
        label.grid(row=0, column=index)


# Defining a function 'next_image' to display the next image in the sequence.
def next_image():
    global current_image_index
    # Increasing the 'current_image_index' to move to the next image.
    current_image_index += 1
    # If 'current_image_index' exceeds the number of captured images, reset it to 0.
    if current_image_index >= len(captured_image_files):
        current_image_index = 0
    # Displaying the image at the updated 'current_image_index'.
    display_image(current_image_index)


# Defining a function 'next_image' to display the next image in the sequence.
def prev_image():
    global current_image_index
    # Increasing the 'current_image_index' to move to the next image.
    current_image_index -= 1
    # Increasing the 'current_image_index' to move to the next image.
    if current_image_index < 0:
        current_image_index = len(captured_image_files) - 1
    # Displaying the image at the updated 'current_image_index'
    display_image(len(os.listdir("captured_objects")) + 1)


# Creating a Tkinter button named 'prev_button' with the text "Previous."
# When this button is clicked, it calls the 'prev_image' function.
prev_button = tk.Button(root, text="Previous", command=prev_image)
prev_button.pack()
# Creating a Tkinter button named 'next_button' with the text "Next."
# When this button is clicked, it calls the 'next_image' function.
next_button = tk.Button(root, text="Next", command=next_image)
next_button.pack()

# Displaying the image at the 'current_image_index' within the sequence of captured images.
display_image(current_image_index)

# Run the GUI
root.mainloop()
