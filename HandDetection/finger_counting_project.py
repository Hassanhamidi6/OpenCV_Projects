import cv2
import mediapipe as mp
import time
import os 
import hand_tracking_module as htm

cap= cv2.VideoCapture(0)
pTime= 0

detector= htm.HandDetector()

folder_path= "fingerImages"
myList= os.listdir(folder_path)
# print(myList)

overlayList= []
for imPath in myList:     # here folder path is fingerprint folder and imPath is image path 
    image= cv2.imread(f"{folder_path}/{imPath}")
    print(f"{folder_path}/{imPath}")
    overlayList.append(image)

tipIds= [4, 8, 12, 16, 20]

while True:
    success , img = cap.read()
    if not success:
        break

    img= detector.findHands(img)
    lmList, bbox= detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) !=0:
        fingers=[]

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]: 
            fingers.append(1)  
        else:
            fingers.append(0)
        # Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]: 
                fingers.append(1)  # because everytime the tip of finger below than 2 or 6 it will not open 
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers= fingers.count(1)
        print(totalFingers)

        overlay = overlayList[totalFingers-1]
        overlay = cv2.resize(overlay, (200, 200))   # resize to fixed size

        h, w, c = overlay.shape
        img[0:h, 0:w] = overlay

        # Number of fingers
        cv2.rectangle(img, (20, 225), (170, 425), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 25)

    cTime= time.time()
    fps = 1/ (cTime - pTime)
    pTime= cTime

    cv2.putText(img, f"FPS {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       

cap.release()
cv2.destroyAllWindows()

     