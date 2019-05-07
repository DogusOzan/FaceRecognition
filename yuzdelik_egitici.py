# train.py
import cv2
import sys
import numpy
import os
import sqlite3


def insertOrUpdate(Id,k,yas,cins):
    conn=sqlite3.connect("YuzdelikBase.db")
    c=conn.cursor()
    c.execute("INSERT INTO people(ID,Name,Age,Cinsiyet) Values(?,?,?,?)",(Id,k,yas,cins))
    conn.commit()
    conn.close()
size = 1
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'


try:
    fn_name = sys.argv[1]
except:
    fn_name=input("Kayıt Edilen Kişi Adı-SoyAdı=>")


kId=input("Kayıt Edilen Kişi id=>")
kYas=input("Kayıt Edilen Kişi Yaşı=>")
kCinsiyet=input("Kayıt Edilen Kişi Cinsiyeti=>")
kadi=fn_name
insertOrUpdate(kId,kadi,kYas,kCinsiyet)
path = os.path.join(fn_dir, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (112, 92)


haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)

# resim dosyası ismi oluşturuyoruz
pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     if n[0]!='.' ]+[0])[-1] + 1

# başlangıç mesajı
print("Bu program yüzünüzün 50 farklı örneğini alıcak. Daha kesin sonuçlar için kafanızı yavaşça hareket ettirin")

# 50 fotoğraf çekene kadar döngüye sokuyoruz
count = 0
pause = 0
count_max = 50
while count < count_max:

    # kamera çalışana kadar döngüye sokuyoruz
    rval = False
    while(not rval):
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    # Get image size
    height, width, channels = frame.shape

    # Flip frame
    frame = cv2.flip(frame, 1, 0)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Scale down for speed
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # Detect faces
    faces = haar_cascade.detectMultiScale(mini)

    # We only consider largest face
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]

        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # Draw rectangle and write name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

        # Remove false positives
        if(w * 6 < width or h * 6 < height):
            print("Algılanan Yüz çok küçük")
        else:

            # To create diversity, only save every fith detected image
            if(pause == 0):
                print("kayıt edilip öğretilen yüz "+str(count+1)+"/"+str(count_max))

                # Save image file
                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
                pin += 1
                count += 1
                pause = 1

    if(pause > 0):
        pause = (pause + 1) % 5
    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if count>50:
        break
