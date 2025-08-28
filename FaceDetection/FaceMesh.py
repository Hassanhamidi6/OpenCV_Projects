import cv2 
import time
import mediapipe as mp

cap= cv2.VideoCapture("vedios\interview.mp4")
pTime=0

mpDraw= mp.solutions.drawing_utils
mpFaceMesh= mp.solutions.face_mesh
faceMesh= mpFaceMesh.FaceMesh(max_num_faces= 1)
drawSpecs= mpDraw.DrawingSpec(thickness= 1, circle_radius=1)  # it is for drawing custom lines on face 

while True:
    success, img= cap.read()
    if not success:
        break
    img = cv2.resize(img, (800, 900))    # Resize the vedio resolution according to my screen

    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img , faceLms, mpFaceMesh.FACEMESH_TESSELATION, 
                                   drawSpecs, drawSpecs)
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y= int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)

    cTime= time.time()
    fps= 1 / (cTime - pTime)
    pTime= cTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break            

cap.release()
cv2.destroyAllWindows()
