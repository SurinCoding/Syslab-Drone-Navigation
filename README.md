# Syslab-Drone-Navigation
Hello everyone. The creators of this project were Surin Wettimuny and Nick Boukens. Our project was on Autonomous Drone Navigation Using Computer Vision and Convolutional Neural Network Navigation. The goal of this project was to address the limitations with the traditional implementation of controlling a drone with a remote controller.

This README.md file will contain all the information needed to actually execute our program on your devices.


## File Descriptions

CNNClassFINAL.py --> This is the class that we made that stores information on the Convolutional Neural Network. This is referred to in FINAL_CV_ML.py

HandsOnlyCode.py --> This file contains single-action recognition mode and hand tracking without the drones. The evaluated frames are outputted onto the screen, allowing you to see what is recognized without observing drone movement.

CNN_FINAL.py --> This was the actual Convolutional Neural Network that generated the weights that our drone uses in the Convolutional Neural Network mode of navigation

FINAL_CV_ML.PY --> This file represents the finished product of our software. It implements all three modes of navigation with drone flight.

MergedDataFinal.csv --> This is a CSV that contains all of the labeled images that we labeled using LabelStudio

CNNweightsFINAL.pth --> This is the actual weights extracted from the Convolutional Neural Network after training it on over 3000 images and 25+ epochs

Training_images_part_aa --> One portion of the training images for our Convolutional Neural Network (needed to split up because it was too big).

Training_images_part_ab --> One portion of the training images for our Convolutional Neural Network (needed to split up because it was too big).

Training_images_part_ac --> One portion of the training images for our Convolutional Neural Network (needed to split up because it was too big).

Training_images_part_ad --> One portion of the training images for our Convolutional Neural Network (needed to split up because it was too big).

Training_images_part_ae --> One portion of the training images for our Convolutional Neural Network (needed to split up because it was too big).

Testing_images --> These are the images that were used to test our Convolutional Neural Network

**Note:** You can download Training_images_part_aa, Training_images_part_ab, Training_images_part_ac, Training_images_part_ad, Training_images_part_ae, and Testing_images at the following GitHub link: https://github.com/NickBoukens/Syslab-Drone-Navigation-Gesture-Based-ML

I cannot attach it here because of the storage capabilities of my computer.

## Package Installations
To run this code, a few packages need to be installed locally on your device. Enter the following commands in a terminal

pip install opencv-python

pip install torch torchvision torchaudio

pip install mediapipe

pip install djitellopy

pip install threading

pip install pillow

Some of these commands may change based on the specifications of your device.

## Code Initialization - Drone Internal Camera Control

This project was created using the DJI Tello as the drone that will fly.

To execute the code, you must make sure that all of the files are in the same directory/folder. Once they are in the same directory/folder, simply open up the directory in an environment like VSCode. Then, turn on the DJI Tello and make sure that your computer is connected to the network that the Tello is emitting. After, you just need to run the following file: FINAL_CV_ML.py

### Single-Action Recognition Mode
After initializing the code in an environment, the drone will enter Single-Action Recognition Mode. 

This mode makes the drone identify hand gestures and execute corresponding actions. This occurs 30 times a second.

The gestures that we coded recognition for are:

Thumbs Up --> Move Up

Thumbs Down --> Move Down

Thumbs Right --> Move Right

Thumbs Left --> Move Left

Open hand, fingers towards the left, fingers parallel, palm facing face --> Move Forward

Open hand, fingers towards the left, fingers parallel, palm facing away from face --> Move Backward

Rockin' Pose --> Change Mode

### Hand Tracking Mode
If the Rockin' Pose is identified in the Single-Action Recognition Mode section, the drone will enter hand tracking mode. In this mode, the drone will follow any three-dimensional movement that your hand performs. This allows for a single person to perform content for a camera without being restricted to staying in one place.

If you want to switch back to Single-Action Recognition Mode, display the Backwards Rockin' Pose, where the palm is facing away from the face

If you want to switch to Convolutional Neural Network Mode, display the Rockin' Pose

### Convolutional Neural Network Mode
If the drone is switched into Convolutional Neural Network Mode, no more human input is needed to control the drone. The drone will recognize a black hoop and evaluate the necessary inputs needed to fly through the hoop. Evaluation of a frame occurs 5 times a second, and an action the drone performs occurs for 0.2 seconds.

##Code Initialization - Phone Internal Camera Control
Our project allows for control using both the DJI Tello's internal camera and a phone camera. To control the drone using the phone camera, an external app needs to be downloaded that allows a phone to transmit video input to a computer. 

For this project, we decided to choose DroidCam. You can install this app on the IOS App Store. It is called DroidCam Webcam & OBS Camera. This method of control requires the use of two network adapters. For our project, we decided to use the TPLink Wireless USB Adapter. Any external network adapter would work for this project. 

First, one network adapter needs to be connected to the same public Wi-Fi network as your phone to enable DroidCam to relay video feed to a computer. Then, the second network adapter needs to connect ot the Wi-Fi network that the DJI Tello emits. Then, all that needs to occur is for you to run FINAL_CV_ML.py. Before you run this code, make sure to find the line of 

camera = cv2.VideoCapture("udp://@0.0.0.0:11111?fifo_size=1000000&overrun_nonfatal=1", cv2.CAP_FFMPEG)

and to change it to 

camera = cv2.VideoCapture(1, cv2.CAP_FFMPEG)

If you want to end the drone program at any time, causing the drone to land in place, simply press Ctrl+C on your device.


### New Purpose of Convolutional Neural Network
For our project, we trained the Convolutional Neural Network on a black hoop that we suspended between two walls using a string and thumbtacks. If a different navigation form is needed for the Convolutional Neural Network mode, training and testing on new images can occur. We trained our images using LabelStudio, which requires you to enter the command

pip install label-studio

into the terminal, and then follow the instructions on this website: https://labelstud.io/learn/getting-started-with-label-studio/

**Note**: Training the Convolutional Neural Network will take multiple hours with 25+ epochs. It took about 6 hours for us to train it.

## If there are any issues that you are facing, you can watch a demonstration video that we put together that documents the processes of each mode. The link to the YouTube video is below.

## https://youtu.be/H7uCPgEFv_o

## If you have any further questions, please reach out to us using any of the following methods.

## Surin Wettimuny; +1 (703) 826-3021; surinw1122@gmail.com

## Nick Boukens; +1 (703) 453-2721; nick.boukens@gmail.com

### Thank you for visiting our project, and we hope you accomplish your task!

