import cv2 
import time
import mediapipe as mp

cap= cv2.VideoCapture(0)

pTime=0

while True:
    success, img= cap.read()
    if not success:
        break
    img = cv2.resize(img, (800, 600))    # Resize the vedio resolution according to my screen

    cTime= time.time()
    fps= 1 / (cTime - pTime)
    pTime= cTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break            

cap.release()
cv2.destroyAllWindows()
