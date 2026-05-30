import cv2
import mediapipe as mp
import time
from djitellopy import tello
import threading
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from CNNClassFINAL import ConvNeuralNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def convertToCNNFrame(frame):
    #change size first
    frame = cv2.resize(frame, (32, 32))# CHECK IF SAME AS CNN INPUT
    frameGreyScaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = np.where(frameGreyScaled < 35, 0, 255).astype(np.uint8)

    greyFrame = np.stack((grey, grey, grey), axis = -1)

    greyFrame = greyFrame/ 255.0 #SCALE
    greyFrame = (greyFrame-0.5)*2
    greyFrame = np.transpose(greyFrame, (2, 0,1))       #first we want width, then height, and then color channel? (CHECK LATER IF IT WORKOS) 

    return torch.tensor(greyFrame, dtype=torch.float32).unsqueeze(0)


ourCN = ConvNeuralNet(numClasses = 9) #Need file called CNNClass.py
ourCN.load_state_dict(torch.load(r"H:\My Drive\Syslab\CNNweightsFINAL.pth", map_location= "cpu00")) #For Future Use, change the pathing to where CNNweightsFINAL is located
ourCN.eval()
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean = [.5, 0.5, 0.5], std = [.5, 0.5, .5])])

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

zDistance = 0




def read_frame(camera):
        global shutdown, latest_frame
        while not shutdown: #turns off the threading
            booleanOfNone, frame = camera.read()
            if booleanOfNone: #not None
                frame = cv2.flip(frame, 1)
                latest_frame = frame

        
def hand_process():
    global latest_frame, results, shutdown
    with mp_hands.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence = 0.6, min_tracking_confidence = 0.6) as hands:
        while not shutdown:
            if latest_frame is None:
                continue
            rgbCopy = cv2.cvtColor(latest_frame.copy(), cv2.COLOR_BGR2RGB)
            results = hands.process(rgbCopy)

def mimic_accel(value, threshold):
    delta = abs(value) - threshold
    max_delta = 0.4
    if delta <= 0:
        return 0
    ratio = min(delta / max_delta, 1.0)

    speed = 15 + ratio * (25)
    return int(speed)


def zmomentum(current, needed):
    value = current - needed
    # print(value)
    # print(int(value*((100))))
    return (int)((value)* -150)
    # else:
        # return (int)((value) * 100)

def main():
    global shutdown, latest_frame, results
    shutdown = False
    latest_frame = None
    results = None

    drone = tello.Tello()
    drone.connect()
    print("Battery level is ", drone.get_battery())
    drone.streamon()
    time.sleep(2)

    drone.takeoff()
    time.sleep(1)
    drone.send_rc_control(0, 0, 40, 0)
    time.sleep(0.5)
    drone.send_rc_control(0, 0, 0, 0)

    camera = cv2.VideoCapture("udp://@0.0.0.0:11111?fifo_size=1000000&overrun_nonfatal=1", cv2.CAP_FFMPEG)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    recognitionmode = 0


    start_time = time.time()
    variableUsedForWhileLoop = True

    thread1 = threading.Thread(target = read_frame, args = (camera, ), daemon = True)
    thread2 = threading.Thread(target = hand_process, daemon = True)

    thread1.start()
    thread2.start()


    while variableUsedForWhileLoop == True:
            

            # if time.time()-start_time > 40: #check if over certaintime limiet
            #     drone.send_rc_control(0,0,0,0)
            
            
            if cv2.waitKey(1) & 0xFF == 27: #27 GIVES ESCAPE
                cv2.destroyAllWindows()
                shutdown = True
                

            if latest_frame is None:
                continue
            
            # in order to see (debug)
            frame_to_DrawOn = latest_frame.copy()
            frame_to_DrawOn = cv2.flip(frame_to_DrawOn, 1)

            cv2.putText(frame_to_DrawOn, str(recognitionmode), (1500, 95), 2, 3, (255, 0, 0), 4)

            if results == None:
                continue #change to hovor later
            

            hand_list = results.multi_hand_landmarks

            if recognitionmode == 2 or hand_list:

                for i in results.multi_hand_landmarks:
                    if recognitionmode == 0: #single gesture action
                        scalingFactor = ((i.landmark[5].y - i.landmark[17].y)**2 + (i.landmark[5].x - i.landmark[17].x)**2)**0.5
                        xDirectionScalingFactor= i.landmark[17].x-i.landmark[5].x


                        mp_drawing.draw_landmarks(frame_to_DrawOn, i, mp_hands.HAND_CONNECTIONS)

                        if ((i.landmark[2].y - i.landmark[4].y)**2 + (i.landmark[2].x - i.landmark[4].x)**2) * 1.33 < ((i.landmark[5].y - i.landmark[17].y)**2 + (i.landmark[5].x - i.landmark[17].x)**2)**0.5:
                            #EITHER A THUMBS UP, THUMBS DOWN, AND THUMBS TO THE SIDE
                            if (i.landmark[2].y-i.landmark[4].y) > scalingFactor/2:
                                if (i.landmark[8].x<i.landmark[6].x and i.landmark[8].x < i.landmark[4].x):
                                    drone.send_rc_control(0, 40, 0, 0) #forward
                                    cv2.putText(frame_to_DrawOn, "Forward", (300,150), 2, 3, (255,0,0), 4)
                                else:
                                    if i.landmark[10].y > i.landmark[4].y:
                                        drone.send_rc_control(0, 0, 30, 0)
                                        cv2.putText(frame_to_DrawOn, "Up", (300,150), 2, 3, (255,0,0), 4)

                            elif (i.landmark[4].y-i.landmark[2].y) > scalingFactor / 2:
                                if (i.landmark[8].x<i.landmark[6].x):
                                    drone.send_rc_control(0, -30, 0, 0)
                                    cv2.putText(frame_to_DrawOn, "Backward", (300,150), 2, 3, (255,0,0), 4)
                                else:
                                    drone.send_rc_control(0, 0, -30, 0)
                                    cv2.putText(frame_to_DrawOn, "Down", (300,150), 2, 3, (255,0,0), 4)
                            elif (i.landmark[1].x - i.landmark[4].x > xDirectionScalingFactor*4/5) and abs((i.landmark[8].y-i.landmark[12].y)) < scalingFactor*2/3:
                                drone.send_rc_control(30, 0, 0, 0)
                                cv2.putText(frame_to_DrawOn, "Right", (300,150), 2, 3, (255,0,0), 4)
                            elif (i.landmark[4].x-i.landmark[1].x) > xDirectionScalingFactor*4/5:
                                drone.send_rc_control(-30, 0, 0, 0)
                                cv2.putText(frame_to_DrawOn, "Left", (300,150), 2, 3, (255,0,0), 4)

                            elif (i.landmark[12].y-i.landmark[8].y)>scalingFactor*2/3:
                                cv2.putText(frame_to_DrawOn, "Change Mode", (300,150), 2, 3, (255,0,0), 4)

                                recognitionmode = 1 
                                zDistance = ((i.landmark[8].x - i.landmark[0].x)**2 + (i.landmark[8].y - i.landmark[0].y)**2)**.5
                                drone.send_rc_control(0,0,0,0)
                                time.sleep(3.5)

                            else:
                                drone.send_rc_control(0, 0, 0, 0)
    
                    elif recognitionmode == 1: #hand tracking
                        # cv2.putText(frame_to_DrawOn, "Go Right", (1200,150), 2, 3, (255,0,0), 4)

                        secondScalingFactor= i.landmark[17].x-i.landmark[5].x
                        currentDistance = ((i.landmark[8].x - i.landmark[0].x)**2 + (i.landmark[8].y - i.landmark[0].y)**2)**.5



                        coordinateKnuckleMiddleBottom = (i.landmark[10].x,i.landmark[10].y)



                        newTuple = (0.5-coordinateKnuckleMiddleBottom[0], 0.5-coordinateKnuckleMiddleBottom[1])
                        speed = 27
                        speedY = 30
                        speedZ = 30
                        xAxis = 0
                        yAxis = 0
                        zAxis = 0
                        if newTuple[0] > 0.15:
                            # if(0.05 < newTuple[1] < 0.1 ):
                            #     cv2.putText(frame_to_DrawOn, "Go UPRIGHT", (600,150), 2, 3, (255,0,0), 4)
                            #     drone.send_rc_control(40, 0, 40, 0)
                            # elif -0.1 < newTuple[1] < -0.05:
                            #     cv2.putText(frame_to_DrawOn, "Go DOWNRIGHT", (600,150), 2, 3, (255,0,0), 4)
                            #     drone.send_rc_control(40, 0, -40, 0)
                            # else: #else go right
                                cv2.putText(frame_to_DrawOn, "Go Right", (1200,150), 2, 3, (255,0,0), 4)
                                # drone.send_rc_control(40, 0, 0, 0)
                                speedX = mimic_accel(newTuple[0], 0.2)
                                xAxis = speedX
                            

                        elif newTuple[0] < -0.15:
                            # if( 0.05 < newTuple[1] < 0.1 ):
                            #     cv2.putText(frame_to_DrawOn, "Go UPLEFT", (600,150), 2, 3, (255,0,0), 4)
                            #     drone.send_rc_control(-40, 0, 40, 0)
                            # elif -0.1 < newTuple[1] < -0.05:
                            #     cv2.putText(frame_to_DrawOn, "Go DOWNLEFT", (600,150), 2, 3, (255,0,0), 4)
                            #     drone.send_rc_control(-40, 0, -40, 0)
                            # else: #else go left
                                cv2.putText(frame_to_DrawOn, "Go Left", (1200,150), 2, 3, (255,0,0), 4)
                                speedX = mimic_accel(newTuple[0], 0.2)
                                xAxis = -1*speedX
                        else:
                            xAxis = 0
                        

                        if zDistance > currentDistance + secondScalingFactor:
                            cv2.putText(frame_to_DrawOn, "Go Forward", (600,400), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0, 40, 0, 0) #forward

                            zSpeed= zmomentum(currentDistance, zDistance)
                            zAxis = zSpeed

                        elif zDistance < currentDistance - secondScalingFactor:
                            cv2.putText(frame_to_DrawOn, "Go Back", (600,400), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0, -40, 0, 0) #backward
                            zSpeed= zmomentum(currentDistance, zDistance)
                            zAxis = zSpeed   
                        else:
                            zAxis = 0
                        if (i.landmark[12].y-i.landmark[8].y)>scalingFactor/3 and ((i.landmark[4].x-i.landmark[2].x) > secondScalingFactor*4):
                            cv2.putText(frame_to_DrawOn, "Change Back", (300,150), 2, 3, (255,0,0), 4)
                            zDistance = ((i.landmark[8].x - i.landmark[0].x)**2 + (i.landmark[8].y - i.landmark[0].y)**2)**.5

                            recognitionmode =0
                            time.sleep(3.5)
                        
                        elif (i.landmark[12].y-i.landmark[8].y)>scalingFactor*2/3:
                            cv2.putText(frame_to_DrawOn, "Change Mode", (300,150), 2, 3, (255,0,0), 4)

                            recognitionmode = 2 
                            zDistance = ((i.landmark[8].x - i.landmark[0].x)**2 + (i.landmark[8].y - i.landmark[0].y)**2)**.5
                            drone.send_rc_control(0,0,0,0)
                            time.sleep(3.5)

                        
                        if newTuple[1] >0.1:
                            cv2.putText(frame_to_DrawOn, "Go Up", (600,150), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0, 0, 40, 0)
                            yAxis = speed


                        elif newTuple[1] < -0.1:
                            cv2.putText(frame_to_DrawOn, "Go Down", (600,150), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0, 0, -40, 0)
                            yAxis = -speed
                        else:
                            yAxis = 0

                        drone.send_rc_control(xAxis, zAxis, yAxis, 0)
                        # else:
                        #     cv2.putText(frame_to_DrawOn, "Stay on Z Axis", (600,400), 2, 3, (255,0,0), 4)


                            # cv2.putText(frame_to_DrawOn, "Go Left", (1200,150), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(-40, 0, 0, 0)

                            
                        # if newTuple[1] >0.1:
                        #     cv2.putText(frame_to_DrawOn, "Go Up", (600,150), 2, 3, (255,0,0), 4)
                        #     drone.send_rc_control(0, 0, 40, 0)


                        # elif newTuple[1] < -0.1:
                        #     cv2.putText(frame_to_DrawOn, "Go Down", (600,150), 2, 3, (255,0,0), 4)
                        #     drone.send_rc_control(0, 0, -40, 0)
                    elif recognitionmode == 2:
                        #CNN OMDE
                        # print("in mode")
                        # tempCopy = latest_frame.copy()

                        newFrame = convertToCNNFrame(latest_frame)

                        with torch.no_grad():

                            result = ourCN(newFrame)

                        aowiejfpioawjefo, valueDirectionPredicted = torch.max(result, 1)
                        numberDirectionPredicted = valueDirectionPredicted.item()

                        newDictForDirections = {0:"Forward", 1:"Left", 2:"Right", 3:"Down", 4:"Up", 5:"Left Up", 6:"Left Down",7:"Right Down", 8:"Right U"}

                        directionPredicted = newDictForDirections[numberDirectionPredicted]
                        probabilitySoftMax = torch.softmax(result, dim = 1) #KLEARNED IN ML

                        confidenceInValue = torch.max(probabilitySoftMax).item()
                        # print(confidenceInValue)
                        if cv2.waitKey(1) & 0xFF == 27:
                            drone.send_rc_control(0,0,0,0)
                            drone.land()
                            break
                        if confidenceInValue > .65:
                            if directionPredicted == "Forward":
                                # print("F")
                                drone.send_rc_control(0,30,0,0)
                                time.sleep(0.2)
                            elif directionPredicted == "Left":
                                drone.send_rc_control(30,0,0,0)
                                # print("L")
                                time.sleep(0.2)
                            elif directionPredicted == "Right":
                                drone.send_rc_control(-30,0,0,0)
                                # print("R")
                                time.sleep(0.2)
                            elif directionPredicted == "Down":
                                drone.send_rc_control(0,0,-45,0)
                                # print("D")
                                time.sleep(0.2)
                            
                            elif directionPredicted == "Up":
                                drone.send_rc_control(0,0,30,0)
                                # print("U")
                                time.sleep(0.2)
                            

                            elif directionPredicted == "Left Up":
                                drone.send_rc_control(30,0,30,0)
                                # print("LU")
                                time.sleep(0.2)


                            elif directionPredicted == "Left Down":
                                drone.send_rc_control(30,0,-30,0)
                                # print("LD")
                                time.sleep(0.2)

                            elif directionPredicted == "Right Down":
                                drone.send_rc_control(-30,0,-30,0)
                                # print("RD")
                                time.sleep(0.2)
                            else:
                                drone.send_rc_control(-30,0,30,0)
                                time.sleep(0.2)
                                # print("RU")
                               
                            #forward second
                            #up third
                            #right first
                        else:
                            #stay still prob maybe contine prev check later
                            drone.send_rc_control(0,0,0,0)
                            time.sleep(0.2)

                            #forward second
                            #up third
                            #right first




            #     for i in hand_list:
            #         mp_drawing.draw_landmarks(frame_to_DrawOn, i, mp_hands.HAND_CONNECTIONS)
            #         if i.landmark[4].y+.1 < i.landmark[5].y:
            # #                 #We are going to say that THUMBS UP is GO UP
            #             drone.move_up(20)
            #             print("Thumbs Up")
                    
            else:
                time.sleep(.1)
                drone.send_rc_control(0, 0, 0, 0)
            # time.sleep(1)
            cv2.imshow("Cam", frame_to_DrawOn)

    shutdown = True
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    
    drone.land()

    drone.end()
    cv2.destroyAllWindows()            

if __name__ == "__main__":

    main()