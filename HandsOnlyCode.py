import cv2
import mediapipe as mp
import time
from djitellopy import tello
import threading

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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
    

def main():
    global shutdown, latest_frame, results
    shutdown = False
    latest_frame = None
    results = None

    # drone = tello.Tello()
    # drone.connect()
    # print("Battery level is ", drone.get_battery())
    # drone.streamon()
    time.sleep(2)

    # drone.takeoff()
    # time.sleep(1)//
    # drone.send_rc_control(0, 0, 40, 0)
    # time.sleep(.5)
    # drone.send_rc_control(0, 0, 0, 0)

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    recognitionmode = 0


    start_time = time.time()
    variableUsedForWhileLoop = True

    thread1 = threading.Thread(target = read_frame, args = (camera, ), daemon = True)
    thread2 = threading.Thread(target = hand_process, daemon = True)

    thread1.start()
    thread2.start()


    while variableUsedForWhileLoop == True:
            # print("DFDF")

            # if time.time()-start_time > 40: #check if over certaintime limiet
            #     drone.send_rc_control(0,0,0,0)
            
            
            if cv2.waitKey(1) & 0xFF == 27: #27 GIVES ESCAPE
                cv2.destroyAllWindows()
                shutdown = True
                break

            if latest_frame is None:
                continue
            
            frame_to_DrawOn = latest_frame.copy()

            cv2.putText(frame_to_DrawOn, str(recognitionmode), (1500, 95), 2, 3, (255, 0, 0), 4)

            if results == None:
                continue #change to hovor later

            hand_list = results.multi_hand_landmarks

            if hand_list:
                # print("DFDF")
                for i in results.multi_hand_landmarks:
                    if recognitionmode == 0: #single gesture action
                        scalingFactor = ((i.landmark[5].y - i.landmark[17].y)**2 + (i.landmark[5].x - i.landmark[17].x)**2)**0.5
                        xDirectionScalingFactor= i.landmark[17].x-i.landmark[5].x


                        mp_drawing.draw_landmarks(frame_to_DrawOn, i, mp_hands.HAND_CONNECTIONS)

                        if ((i.landmark[2].y - i.landmark[4].y)**2 + (i.landmark[2].x - i.landmark[4].x)**2) * 1.33 < ((i.landmark[5].y - i.landmark[17].y)**2 + (i.landmark[5].x - i.landmark[17].x)**2)**0.5:
                            #EITHER A THUMBS UP, THUMBS DOWN, AND THUMBS TO THE SIDE
                            if (i.landmark[2].y-i.landmark[4].y) > scalingFactor/2:
                                if (i.landmark[8].x<i.landmark[6].x and i.landmark[8].x < i.landmark[4].x):
                                    # drone.send_rc_control(0, 40, 0, 0) #forward
                                    cv2.putText(frame_to_DrawOn, "Forward", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)
                                else:
                                    if i.landmark[10].y > i.landmark[4].y:
                                        # drone.send_rc_control(0, 0, 40, 0)
                                        cv2.putText(frame_to_DrawOn, "Up", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)

                            elif (i.landmark[4].y-i.landmark[2].y) > scalingFactor / 2:
                                if (i.landmark[8].x<i.landmark[6].x):
                                    # drone.send_rc_control(0, -40, 0, 0)
                                    cv2.putText(frame_to_DrawOn, "Backward", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)
                                else:
                                    # drone.send_rc_control(0, 0, -40, 0)
                                    cv2.putText(frame_to_DrawOn, "Down", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)
                            elif (i.landmark[1].x - i.landmark[4].x > xDirectionScalingFactor*(4/5)) and abs((i.landmark[8].y-i.landmark[12].y)) < scalingFactor*2/3:
                                # drone.send_rc_control(40, 0, 0, 0)
                                cv2.putText(frame_to_DrawOn, "Right", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)
                            elif (i.landmark[4].x-i.landmark[1].x) > xDirectionScalingFactor*0.8:
                                # drone.send_rc_control(-40, 0, 0, 0)
                                cv2.putText(frame_to_DrawOn, "Left", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)

                            elif (i.landmark[12].y-i.landmark[8].y)>scalingFactor*2/3:
                                cv2.putText(frame_to_DrawOn, "Change Mode", (100,300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 20)

                                recognitionmode = 1 
                                zDistance = ((i.landmark[8].x - i.landmark[0].x)**2 + (i.landmark[8].y - i.landmark[0].y)**2)**.5
                                # drone.send_rc_control(0,0,0,0)
                                time.sleep(0.5)

                            # else:
                                # drone.send_rc_control(0, 0, 0, 0)
    
                    elif recognitionmode == 1: #hand tracking
                        # cv2.putText(frame_to_DrawOn, "Go Right", (1200,150), 2, 3, (255,0,0), 4)




                        coordinateKnuckleMiddleBottom = (i.landmark[10].x,i.landmark[10].y)

                        secondScalingFactor= i.landmark[17].x-i.landmark[5].x



                        currentDistance = ((i.landmark[8].x - i.landmark[0].x)**2 + (i.landmark[8].y - i.landmark[0].y)**2)**.5

                        newTuple = (0.5-coordinateKnuckleMiddleBottom[0], 0.5-coordinateKnuckleMiddleBottom[1])

                        if newTuple[0] >0.1:
                            cv2.putText(frame_to_DrawOn, "Go Right", (1200,150), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(40, 0, 0, 0)

                                

                        elif newTuple[0] < -0.1:
                            cv2.putText(frame_to_DrawOn, "Go Left", (1200,150), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(-40, 0, 0, 0)

                            
                        if newTuple[1] >0.1:
                            cv2.putText(frame_to_DrawOn, "Go Up", (600,150), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0, 0, 40, 0)


                        elif newTuple[1] < -0.1:
                            cv2.putText(frame_to_DrawOn, "Go Down", (600,150), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0, 0, -40, 0)

                        # else:
                        #     drone.send_rc_control(0, 0, 0, 0)

                        elif zDistance > currentDistance + secondScalingFactor:
                            cv2.putText(frame_to_DrawOn, "Go Forward", (600,400), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0, 40, 0, 0)

                        elif zDistance < currentDistance - secondScalingFactor:
                            cv2.putText(frame_to_DrawOn, "Go Back", (600,400), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0, -40, 0, 0)

                        else:
                            cv2.putText(frame_to_DrawOn, "Stay on Z Axis", (600,400), 2, 3, (255,0,0), 4)
                            # drone.send_rc_control(0,0,0,0)

            #     for i in hand_list:
            #         mp_drawing.draw_landmarks(frame_to_DrawOn, i, mp_hands.HAND_CONNECTIONS)
            #         if i.landmark[4].y+.1 < i.landmark[5].y:
            # #                 # are going to say that THUMBS UP is GO UP
            #             drone.move_up(20)
            #             print("Thumbs Up")
                    
            else:
                time.sleep(.1)
                # drone.send_rc_control(0, 0, 0, 0)
            # time.sleep(1)
            cv2.imshow("Cam", frame_to_DrawOn)

    shutdown = True
    # drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    
    # drone.land()

    # drone.end()
    cv2.destroyAllWindows()            

if __name__ == "__main__":
    main()

