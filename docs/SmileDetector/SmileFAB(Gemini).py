import cv2

# 1. Բեռնում ենք դեմքի և ժպիտի հայտնաբերման մոդելները (Haar Cascades)
# cv2.data.haarcascades-ը ավտոմատ գտնում է OpenCV-ի մեջ ներկառուցված մոդելների ֆայլերը
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# 2. Միացնում ենք վեբ-տեսախցիկը
# '0'-ն նշանակում է համակարգչի հիմնական (կառուցված) տեսախցիկը:
# Եթե ունեք արտաքին տեսախցիկ, կարող եք փորձել '1' կամ '2'
cap = cv2.VideoCapture(0)

print("Ծրագիրը միացված է: Սեղմեք 'q' դուրս գալու համար:")

while True:
    # 3. Կարդում ենք տեսախցիկի ընթացիկ կադրը (frame)
    ret, frame = cap.read()
    
    # Եթե կադրը չի կարդացվել, դուրս ենք գալիս ցիկլից
    if not ret:
        print("Սխալ. Չհաջողվեց կարդալ տեսախցիկի պատկերը:")
        break

    # 4. Պատկերը դարձնում ենք սևուսպիտակ (Grayscale)
    # Պատճառ: Հայտնաբերման մոդելներն ավելի արագ և ճշգրիտ են աշխատում սևուսպիտակ գույներով
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Գտնում ենք դեմքերը սևուսպիտակ պատկերի վրա
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 6. Անցնում ենք հայտնաբերված յուրաքանչյուր դեմքի վրայով
    # x, y - դեմքի վերևի ձախ անկյան կոորդինատներն են
    # w, h - դեմքի լայնությունն ու բարձրությունն են
    for (x, y, w, h) in faces:
        
        # Գծում ենք կապույտ ուղղանկյուն դեմքի շուրջը (BGR ձևաչափով՝ (255, 0, 0))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 7. Սահմանում ենք «Հետաքրքրության տիրույթ» (ROI - Region of Interest)
        # Քանի որ ժպիտը գտնվում է դեմքի վրա, մենք ժպիտը կփնտրենք միայն դեմքի հատվածում՝ խնայելով ռեսուրսներ:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 8. Փնտրում ենք ժպիտը դեմքի սևուսպիտակ տիրույթում
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        # 9. Անցնում ենք հայտնաբերված յուրաքանչյուր ժպիտի վրայով
        for (sx, sy, sw, sh) in smiles:
            # Գծում ենք կանաչ ուղղանկյուն ժպիտի շուրջը (0, 255, 0)
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            
            # Էկրանին ավելացնում ենք "Smiling" տեքստը, երբ ժպիտ է գտնվում
            cv2.putText(frame, 'Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 10. Ցուցադրում ենք վերջնական կադրը նոր պատուհանում
    cv2.imshow('Smile Detector', frame)

    # 11. Սպասում ենք ստեղնաշարի հրահանգին (1 միլիվայրկյան)
    # Եթե սեղմվում է 'q' տառը, ընդհատում ենք ցիկլը
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 12. Ծրագրի ավարտին անջատում ենք տեսախցիկը և փակում բոլոր պատուհանները
cap.release()
cv2.destroyAllWindows()
