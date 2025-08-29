import cv2 
import time 
import mediapipe as mp 


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon= minDetectionCon

        self.mpFaceDetection= mp.solutions.face_detection
        self.mpDraw= mp.solutions.drawing_utils
        self.faceDetection= self.mpFaceDetection.FaceDetection()

    def findFaces(self, img, draw=True):

        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results= self.faceDetection.process(imgRGB)
        # print(results)
     
        bboxs=[]
        if self.results.detections:
            for id, detection  in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection) # It give us the bounding box and coordinates by deafault   
                # print(id, detections)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)

                bboxC= detection.location_data.relative_bounding_box
                ih, iw, ic= img.shape       # this is img shope, height and 
                bbox= int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img= self.fancyDraw(img, bbox)
                    # Here we are writing the text on the bounding box
                    cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20), 
                                cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)
        return img, bboxs
    
    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):     # l= length , t= thickness . rt=  rectangle thickness
        x, y,w,h = bbox
        x1, y1, = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255),  rt)

        # Top left x,y
        cv2.line(img, (x,y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x,y), (x, y+l), (255, 0, 255), t)
        # Top Right x1,y
        cv2.line(img, (x1,y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1,y), (x1, y+l), (255, 0, 255), t)
        # Bottom left x,y1
        cv2.line(img, (x,y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x,y1), (x, y1-l), (255, 0, 255), t)
        # Bottom Right x1,y
        cv2.line(img, (x1,y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1,y1), (x1, y1-l), (255, 0, 255), t)
        
        return img


def main():
        
    cap= cv2.VideoCapture("vedios\man_on_call.mp4")

    detector= FaceDetector()   # class name  

    pTime= 0


    while True:
        success, img= cap.read()
        if not success:
            break 
        img = cv2.resize(img, (800, 600))    # Resize the vedio resolution according to my screen

        img, bboxs = detector.findFaces(img)
        print(bboxs)

        cTime= time.time()
        fps= 1 /(cTime - pTime)
        pTime=cTime

        cv2.putText(img, f"FPS: {str(int(fps))}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()