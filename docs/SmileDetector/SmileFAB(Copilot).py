# SmileFAB(Copilot).py
import cv2
import sys

# --- Cascade ֆայլերի ուղիները ---
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
SMILE_CASCADE_PATH = "haarcascade_smile.xml"

# --- Բեռնում ---
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)

if face_cascade.empty():
    print("Error: cannot load face cascade")
    sys.exit(1)

if smile_cascade.empty():
    print("Error: cannot load smile cascade")
    sys.exit(1)

# --- Webcam բացում ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open webcam")
    sys.exit(1)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Դեմքի հայտնաբերում
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_roi_gray = gray[y:y + h, x:x + w]
        face_roi_color = frame[y:y + h, x:x + w]

        # Ժպիտի հայտնաբերում
        smiles = smile_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25)
        )

        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Ցանկության դեպքում կարող եք նաև շրջանակել ժպիտը
            sx, sy, sw, sh = smiles[0]
            cv2.rectangle(face_roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Smiling", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Smile Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
