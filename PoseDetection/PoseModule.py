import cv2
import mediapipe as mp  
import time


class poseDetector():
    def __init__(self, mode=False, modelComplexity=1, smooth=True, 
                 detectionCon=0.5, trackingCon=0.5):
        
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                    model_complexity=self.modelComplexity,
                                    smooth_landmarks=self.smooth,
                                    min_detection_confidence=self.detectionCon,
                                    min_tracking_confidence=self.trackingCon)
        
    def findPose(self, img, draw=True):
        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # here we change the color to RGB
        self.results= self.pose.process(imgRGB)
        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                #  Here we have landmarks and connection betwen them 
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
                                            self.mpPose.POSE_CONNECTIONS) 
                
        return img

    def findposition(self, img, draw=True):
        lmlist=[]                            # list for landmarks
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c= img.shape       # hieght , width and channel
                # print(id, lm)

                cx, cy= int(lm.x*w) , int(lm.y*h)
                lmlist.append([id, cx, cy])    # here we print only id and x and y coordiantes
                if draw:
                    cv2.circle(img, (cx,cy), 3, (0,255,0), cv2.FILLED)
        
        return lmlist

def main():
    cap=cv2.VideoCapture("vedios\ManWithRope.mp4")  # vedio here

    pTime= 0     # Previous Time
    detector= poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.resize(img, (800, 600))    # Resize the vedio resolution according to my screen

        img= detector.findPose(img)
        lmlist= detector.findposition(img, draw=False)
        print(lmlist[14])
        cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 3, (0,255,0), cv2.FILLED)

        cTime= time.time()     # Current time 
        fps= 1 / (cTime - pTime)
        pTime= cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        cv2.imshow("Image", img)        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
     
        
if __name__ == "__main__":
    main()