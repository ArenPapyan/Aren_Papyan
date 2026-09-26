import cv2


# ============================================================
# 1. Բեռնում ենք դեմքի և ժպիտի հայտնաբերման պատրաստի մոդելները
# ============================================================

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
smile_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Ստուգում ենք, որ մոդելները ճիշտ բեռնվել են
if face_cascade.empty():
    raise RuntimeError("Չհաջողվեց բեռնել դեմքի հայտնաբերման մոդելը։")

if smile_cascade.empty():
    raise RuntimeError("Չհաջողվեց բեռնել ժպիտի հայտնաբերման մոդելը։")


# ============================================================
# 2. Միացնում ենք համակարգչի տեսախցիկը
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Տեսախցիկը չի բացվում։ Ստուգիր, արդյոք այն հասանելի է։")


# Այս փոփոխականը հաշվում է ժպիտի հայտնաբերմամբ կադրերը
smile_frames = 0


# ============================================================
# 3. Հիմնական ցիկլը
# ============================================================

while True:

    # Կարդում ենք տեսախցիկի հերթական կադրը
    success, frame = camera.read()

    if not success:
        print("Չհաջողվեց ստանալ տեսախցիկի կադրը։")
        break

    # Հայելային պատկեր
    frame = cv2.flip(frame, 1)

    # ========================================================
    # 4. Կադրը դարձնում ենք սև-սպիտակ
    # ========================================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # 5. Հայտնաբերում ենք դեմքերը
    # ========================================================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    smile_detected = False

    # ========================================================
    # 6. Յուրաքանչյուր դեմքի վրա փնտրում ենք ժպիտ
    # ========================================================

    for (x, y, w, h) in faces:

        # Դեմքի հատվածը առանձնացնում ենք
        face_gray = gray[y:y + h, x:x + w]

        # Նույն հատվածը վերցնում ենք նաև գունավոր նկարից
        face_color = frame[y:y + h, x:x + w]

        # Դեմքի հատվածում փնտրում ենք ժպիտ
        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(30, 30)
        )

        # Եթե ժպիտ է հայտնաբերվել
        if len(smiles) > 0:
            smile_frames += 1
        else:
            smile_frames = max(0, smile_frames - 1)

        # Մի քանի հաջորդական կադրից հետո հաստատում ենք ժպիտը
        if smile_frames >= 3:
            smile_detected = True

        # Դեմքի շուրջը ուղղանկյուն
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Գրում ենք արդյունքը
        if smile_frames >= 3:
            text = "SMILE DETECTED"
        else:
            text = "NO SMILE"

        cv2.putText(
            frame,
            text,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if smile_frames >= 3 else (0, 0, 255),
            2
        )

        # Ժպիտի հայտնաբերված հատվածը նշում ենք
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(
                face_color,
                (sx, sy),
                (sx + sw, sy + sh),
                (0, 255, 255),
                2
            )

    # ========================================================
    # 7. Ընդհանուր արդյունքը
    # ========================================================

    if smile_detected:
        status = "ԺՊԻՏԸ ՀԱՅՏՆԱԲԵՐՎԵՑ"
        status_color = (0, 255, 0)
    else:
        status = "ԺՊԻՏ ՉԻ ՀԱՅՏՆԱԲԵՐՎԵԼ"
        status_color = (0, 0, 255)

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )

    # Ցուցադրում ենք տեսախցիկի պատկերը
    cv2.imshow("Smile Detector", frame)

    # q ստեղնով դուրս ենք գալիս
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# 8. Փակում ենք տեսախցիկը
# ============================================================

camera.release()
cv2.destroyAllWindows()
