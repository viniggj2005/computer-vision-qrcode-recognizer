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
            resultados = modelo(frame, verbose=False)
        
        for box in resultados[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (3, 120, 255), 2)
            cv2.putText(frame, 'QRCode', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        cv2.imshow("Qrcode recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
