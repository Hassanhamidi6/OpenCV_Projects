import cv2
import mediapipe as mp
import time
import PoseModule as pm 


cap=cv2.VideoCapture("vedios\ManWithRope.mp4")  # vedio here

pTime= 0     # Previous Time

detector= pm.poseDetector()

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (800, 600))    # Resize the vedio resolution according to my screen

    img= detector.findPose(img)
    lmlist= detector.findposition(img, draw=False)
    if len(lmlist) != 0:
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 5, (0,255,0), cv2.FILLED)

    cTime= time.time()     # Current time 
    fps= 1 / (cTime - pTime)
    pTime= cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image", img)    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    