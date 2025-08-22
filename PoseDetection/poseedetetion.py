import cv2
import mediapipe as mp  
import time


cap=cv2.VideoCapture("vedios\ManWithRope.mp4")  # vedio here

mpDraw= mp.solutions.drawing_utils 
mpPose = mp.solutions.pose
pose = mpPose.Pose()

pTime= 0     # Previous Time

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (800, 600))    # Resize the vedio resolution according to my screen

    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # here we change the color to RGB
    results= pose.process(imgRGB)
    # print(results.pose_landmarks) # this gives the landmarks as well as X and Y coordinates

    if results.pose_landmarks:
        #  Here we have landmarks and connection betwen them 
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) 
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c= img.shape       # hieght , width and channel
            print(id, lm)

            cx, cy= int(lm.x*w) , int(lm.y*h)
            cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)


    cTime= time.time()     # Current time 
    fps= 1 / (cTime - pTime)
    pTime= cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image", img)    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 