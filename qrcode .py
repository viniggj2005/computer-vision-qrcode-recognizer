import cv2
import os
import contextlib
from ultralytics import YOLO

modelo = YOLO('./qrcode.pt') 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            resultados = modelo(frame,verbose=False)

        frame_anotado = resultados[0].plot()

        cv2.imshow("Video", frame_anotado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
