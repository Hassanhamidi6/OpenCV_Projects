import cv2
import mediapipe as mp 
import hand_tracking_module as htm
import time
import math
import numpy as np 

cap = cv2.VideoCapture(0)
pTime=0

detector= htm.HandDetector()

# ----- I copied this code from the pycaw github repo --------------

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

device= AudioUtilities.GetSpeakers()
interface= device.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume= cast(interface, POINTER(IAudioEndpointVolume))
VolRange= volume.GetVolumeRange()

# ----------------

minVol= VolRange[0]
maxVol= VolRange[1] 

vol= 0
volBar= 400
volPer= 0 

while True:
    success, img = cap.read()
    if not success:
        break 

    img= detector.findHands(img)
    lmList, bbox= detector.findPosition(img, draw=False) # it is use to find the position of hands 
   
    if len(lmList) !=0:
        # print(lmList[4], lmList[8])  # here we are printing tips of thumb and index finger

        x1, y1= lmList[4][1], lmList[4][2]
        x2, y2= lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2) // 2, (y1+y2) // 2   # here we check the centre of the line 

        cv2.circle(img, (x1, y1), 10, (255,0,255), cv2.FILLED) # to make circles
        cv2.circle(img, (x2, y2), 10, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255,0,255), 3)
        cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED) # making circle in the centre of the line 

        length= math.hypot(x2-x1, y2-y1)
        # print(length)

        # volume Range -65 -> 0
        # Hand  Range 50 -> 300 

        vol= np.interp(length, [50, 300], [minVol,maxVol]) # we can change these values to make it more smooth
        volBar= np.interp(length, [50, 300], [400,150]) 
        volPer= np.interp(length, [50, 300], [0,100]) 
        print(int(length), int(vol))

        volume.SetMasterVolumeLevelScalar(volPer/100,None) 

        if length<50:
            cv2.circle(img, (cx, cy), 10, (0,255,0), cv2.FILLED) 
    
    # making volume UI
    cv2.rectangle(img, (50, 150), (85, 400), (0,255,0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0,255,0), cv2.FILLED)
    cv2.putText(img, f"Volume: {int(volPer)}%", (30, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)



    cTime= time.time()
    fps= 1 / (cTime - pTime)
    pTime=cTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break            

cap.release()
cv2.destroyAllWindows()

