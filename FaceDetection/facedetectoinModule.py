import cv2 
import time
import mediapipe as mp


class FaceDetection():
    def __init__(self, staticMode=False, maxFaces= 2, refine_landmarks=False, minDetectionConf=0.5, minTrackConf=0.5 ):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.refine_landmarks=refine_landmarks
        self.minDetectionConf=minDetectionConf
        self.minTrackConf=minTrackConf

        self.mpDraw= mp.solutions.drawing_utils
        self.mpFaceMesh= mp.solutions.face_mesh
        self.faceMesh= self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, refine_landmarks, self.minDetectionConf, self.minTrackConf)
        self.drawSpecs= self.mpDraw.DrawingSpec(thickness= 1, circle_radius=1)  # it is for drawing custom lines on face 


    def findFaceMesh(self, img, draw= True):
        self.imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results= self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, 
                                        self.drawSpecs, self.drawSpecs)
                    # for id, lm in enumerate(faceLms.landmark):
                    #     # print(lm)
                    #     ih, iw, ic = img.shape
                    #     x, y= int(lm.x*iw), int(lm.y*ih)
                    #     print(id, x, y)

        return img 


def main():
    cap= cv2.VideoCapture("vedios\podcast.mp4")
    pTime=0

    detector = FaceDetection()

    while True:
        success, img= cap.read()
        if not success:
            break
        img = cv2.resize(img, (800, 900))    # Resize the vedio resolution according to my screen

        img = detector.findFaceMesh(img ,False)
        
        cTime= time.time()
        fps= 1 / (cTime - pTime)
        pTime= cTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            

    cap.release()
    cv2.destroyAllWindows()




if __name__ =="__main__":
    main()