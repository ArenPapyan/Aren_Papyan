import cv2
import sys

# 1. Բեռնում ենք դեմքի և ժպիտի Haar cascade մոդելները
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# Ստուգում ենք՝ արդյոք մոդելները բեռնվել են
if face_cascade.empty():
    raise IOError("Չհաջողվեց բեռնել դեմքի cascade ֆայլը")

if smile_cascade.empty():
    raise IOError("Չհաջողվեց բեռնել ժպիտի cascade ֆայլը")

# 2. Բացում ենք տեսախցիկը
# 0 նշանակում է համակարգի առաջին տեսախցիկը
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Չհաջողվեց բացել տեսախցիկը")
    sys.exit()

while True:
    # 3. Կարդում ենք տեսախցիկի մեկ կադր
    ret, frame = cap.read()

    if not ret:
        print("Կադրը չհաջողվեց կարդալ")
        break

    # 4. Պատկերը փոխարկում ենք մոխրագույնի
    # Haar cascade-ները աշխատում են գորշ պատկերով
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Հայտնաբերում ենք դեմքերը
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 6. Անցնում ենք հայտնաբերված բոլոր դեմքերով
    for (x, y, w, h) in faces:
        # Դեմքի շուրջ կապույտ շրջանակ
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Դեմքի հատվածը առանձնացնում ենք
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # 7. Դեմքի ներսում փնտրում ենք ժպիտ
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25)
        )

        # 8. Եթե ժպիտ է հայտնաբերվել, գծում ենք կանաչ շրջանակ և գրում Smile
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(
                roi_color,
                (sx, sy),
                (sx + sw, sy + sh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Smile",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    # 9. Ցուցադրում ենք արդյունքը
    cv2.imshow("Smile Detector", frame)

    # 10. Եթե սեղմենք q, ծրագիրը կդադարի
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 11. Ազատում ենք ռեսուրսները
cap.release()
cv2.destroyAllWindows()
