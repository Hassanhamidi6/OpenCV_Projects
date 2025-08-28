import cv2
import mediapipe as mp
import time 

cap= cv2.VideoCapture(0)
pTime=0

mpDraw= mp.solutions.drawing_utils
mpHands= mp.solutions.hands 
hands = mpHands.Hands()

while True:
    success, img  =cap.read()
    if not success:
        break 
    
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                w, h, c  =img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                # cv2.circle(img, (cx, cy), 15, (0,255,0), cv2.FILLED)
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
    


    cTime = time.time()
    fps= 1 /(cTime - pTime)
    pTime=cTime

    cv2.putText(img, f"FPS {int(fps)}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       

cap.release()
cv2.destroyAllWindows()

