import cv2 
import time 
import mediapipe as mp 

cap= cv2.VideoCapture(0)

mpFaceDetection= mp.solutions.face_detection
mpDraw= mp.solutions.drawing_utils
faceDetection= mpFaceDetection.FaceDetection()

pTime= 0

while True:
    success, img= cap.read()
    if not success:
        break 
    img = cv2.resize(img, (1000, 600))    # Resize the vedio resolution according to my screen

    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection  in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection) # It give us the bounding box and coordinates by deafault   

            # print(id, detections)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC= detection.location_data.relative_bounding_box
            ih, iw, ic= img.shape       # this is img shope, height and 
            bbox= int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih) 
            cv2.rectangle(img, bbox, (255, 0, 255),  2)

            # Here we are writing the text on the bounding box
            cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-20), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)


    cTime= time.time()
    fps= 1 /(cTime - pTime)
    pTime=cTime

    cv2.putText(img, f"FPS: {str(int(fps))}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
        
