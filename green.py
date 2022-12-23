"""Prueba de calibración del color verde"""

# Importación de los módulos necesarios
import os
import glob
import cv2 as cv
import numpy as np
import smtplib, ssl
from datetime import date, datetime
from collections import Counter
from numpy.lib.function_base import average

# Input de usuario para la decisión de si se va a proceder a entrenamiento o validación
testType = input("Indroduza el tipo de test (e - entrenamiento | v - validación): ")

start_time = datetime.now()

if testType == "e":
  src = "Entrada/eVerde"
  pathTemp = "eVerde"
elif testType == "v":
  src = "Entrada/vVerde"
  pathTemp = "vVerde"
else:
  TypeError
  pass

# Parámetros a iterar
area = np.arange(150, 200, 50)
detection = np.arange(0, 0.1, 0.1)
ratio = np.arange(0.8, 0.9, 0.1)
jump = np.arange(18, 20, 2)

hueMin = np.arange(40, 41, 1)
saturationMin = np.arange(60, 61, 1)
valueMin = np.arange(0, 1, 1)
hueMax = np.arange(80, 81, 1)
saturationMax = np.arange(255, 256, 1)
valueMax = np.arange(255, 256, 1)

error = []
coor = []
names = []

iter = len(area)*len(detection)*len(ratio)*len(jump)*len(hueMin)*len(hueMax)*len(saturationMin)*len(saturationMax)*len(valueMin)*len(valueMax)*len(os.listdir(f"{src}"))

pathCurr = os.getcwd() # El directorio actual debe ser /Parent
print("El directorio actual es %s" % pathCurr)

# Creación del directorio donde guardar las imágenes producidas
os.chdir("Resultados") # Bajada a /Resultados
cwd = os.getcwd()
print("El directorio de salida será %s" % cwd)

try:
  os.mkdir(pathTemp) # Se crea el directorio de salida
except OSError:
  print("La generación del directorio %s falló" % pathTemp)
else:
  print("El directorio %s ha sido creado con éxito" % pathTemp)

# Función de análisis de color
def green(img):
        image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        detection = img.copy()

        green_lower_lower = np.array([hmin, smin, vmin])
        green_lower_upper = np.array([hmax, smax, vmax])

        mask_green1 = cv.inRange(image, green_lower_lower, green_lower_upper)

        green_upper_lower = np.array([hmin, smin, vmin])
        green_upper_upper = np.array([hmax, smax, vmax])

        mask_green2 = cv.inRange(image, green_upper_lower, green_upper_upper)
        mask_green = mask_green1 + mask_green2

        detection_green = cv.bitwise_and(detection, detection, mask=mask_green)
        green_ratio = (cv.countNonZero(mask_green))/(img.size/3)
        green_ratio_rounded = np.round(green_ratio * 100, 2)
        if green_ratio_rounded >= d:

            img_contour = detection_green.copy()
            img_gray = cv.cvtColor(detection_green, cv.COLOR_BGR2GRAY)
            img_blur = cv.GaussianBlur(img_gray, (7, 7), 1, dst=250, sigmaY=0, borderType=1)
            img_blank = np.zeros_like(img)

            ret, thresh = cv.threshold(img_blur, 1, 255, cv.THRESH_BINARY)

            global yGreen, x_green, y_green
            yGreen = []
            x_green = []
            y_green = []
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv.contourArea(contour)
                if area > a:
                    cv.drawContours(img_contour, contour, -1, (255, 10, 0), 2)
                    perimeter = cv.arcLength(contour, True)
                    corners = cv.approxPolyDP(contour, 0.015 * perimeter, True)
                    x, y, width, height = cv.boundingRect(corners)
                    if width/height >= r:
                        yGreen.append(y)
                        corners_filtered = [x, y, x+width, y+height]
                        x1 = corners_filtered[0]
                        y1 = corners_filtered[1]
                        x2 = corners_filtered[2]
                        y2 = corners_filtered[3]
                        y_green.append(y1)
                        x_green.append(x1)
                        y_green.append(y2)
                        y_green.append(x2)

                        cv.rectangle(img_contour, (x, y), (x+width, y+height), (0, 255, 0), 2)
            yGreen.reverse()
            y_green.reverse()
            x_green.reverse()

            if yGreen:
                jump = j
                indices = [i + 1 for (x, y, i) in zip(yGreen, yGreen[1:], range(len(yGreen))) if jump < abs(x - y)]
                start = 0
                end = len(yGreen)
                result = [yGreen[start:end] for start, end in zip([0] + indices, indices + [len(yGreen)])]

                yGreen = []
                for i in result:
                    yGreen.append(average(i))

                if len(yGreen) == 0:
                    yGreen.extend([0, 0, 0, 0, 0])
                elif len(yGreen) == 1:
                    yGreen.extend([0, 0, 0, 0])
                elif len(yGreen) == 2:
                    yGreen.extend([0, 0, 0])
                elif len(yGreen) == 3:
                    yGreen.extend([0, 0])
                elif len(yGreen) == 4:
                    yGreen.extend([0])
                elif len(yGreen) == 5:
                    pass
            elif not yGreen:
                yGreen = [0, 0, 0, 0, 0]

            os.chdir("Resultados")
            cv.imwrite(f"{pathTemp}/verde_{name}.jpg", img_contour)
            os.chdir(pathParent)

        else:
            yGreen = [0, 0, 0, 0, 0]
            x_green = []
            y_green = []
    
        def checker(yGreen, name, names, error, coor, iter):
            
            names.append(name)
            name = list(name)

            yGreen = [i for i in yGreen if i != 0]
            
            if name.count("5") == len(yGreen):
                error.append(1)
            elif name.count("5") != len(yGreen):
                error.append(0)
            else:
                print(NameError)
            
            coor.append([a, d, r, j, hmin, smin, vmin, hmax, smax, vmax])

            mark = np.round(len(error)*(100/iter), 10)
            print(mark,"%")
            passes = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            for i in passes:
                if i == mark:
                    print(i,"%")
                else:
                    pass

            return error, coor, names

        checker(yGreen, name, names, error, coor, iter)

# Preparación del entorno para el loop de análisis
pathParent = os.path.dirname(os.getcwd()) # Definición del directorio /Parent
os.chdir(pathParent) # Subida a /Parent

# Entrada en el loop, ahora mismo debemos estar en /Parent
for image in glob.glob(f"{src}/*.jpg"):
    img = cv.imread(image)
    name = os.path.splitext(os.path.basename(f"{src}/{image}"))[0]

    for a in area:
        for d in detection:
            for r in ratio:
                for j in jump:
                    for hmin in hueMin:
                        for smin in saturationMin:
                            for vmin in valueMin:
                                for hmax in hueMax:
                                    for smax in saturationMax:
                                        for vmax in valueMax:

                                            green(img)

cv.waitKey(0)

cv.destroyAllWindows()

def condition(x): 
    return x == 1.0

def most_frequent(list):
    return max(set(list), key = list.count)

output = [i for i, element in enumerate(error) if condition(element)]

params = [coor[x] for x in output]

params_a = list(set([i[0] for i in params]))
params_d = list(set([i[1] for i in params]))
params_r = list(set([i[2] for i in params]))
params_j = list(set([i[3] for i in params]))
params_hmin = list(set([i[4] for i in params]))
params_smin = list(set([i[5] for i in params]))
params_vmin = list(set([i[6] for i in params]))
params_hmax = list(set([i[7] for i in params]))
params_smax = list(set([i[8] for i in params]))
params_vmax = list(set([i[9] for i in params]))

list_a = [i[0] for i in params]
list_d = [i[1] for i in params]
list_r = [i[2] for i in params]
list_j = [i[3] for i in params]
list_hmin = [i[4] for i in params]
list_smin = [i[5] for i in params]
list_vmin = [i[6] for i in params]
list_hmax = [i[7] for i in params]
list_smax = [i[8] for i in params]
list_vmax = [i[9] for i in params]

repetitions_a = dict(Counter(list_a))
repetitions_d = dict(Counter(list_d))
repetitions_r = dict(Counter(list_r))
repetitions_j = dict(Counter(list_j))
repetitions_hmin = dict(Counter(list_hmin))
repetitions_smin = dict(Counter(list_smin))
repetitions_vmin = dict(Counter(list_vmin))
repetitions_hmax = dict(Counter(list_hmax))
repetitions_smax = dict(Counter(list_smax))
repetitions_vmax = dict(Counter(list_vmax))


hits = [names[x] for x in output]
true_hits = list(dict.fromkeys(hits))
eficaciaReal = np.round((len(true_hits) / len(os.listdir(f"{src}"))*100), 2)
efiaciaIteraciones = np.round(len(output)/len(error)*100, 2)

# print("Áreas con aciertos:", params_a)
# print("Ratios de detección con aciertos:", params_d)
# print("Ratios alto ancho con aciertos:", params_r)
# print("Saltos con aciertos:", params_j)
# print("Hmin con aciertos:", params_hmin)
# print("Smin de detección con aciertos:", params_smin)
# print("Vmin alto ancho con aciertos:", params_vmin)
# print("Hmax con aciertos:", params_hmax)
# print("Smax con aciertos:", params_smax)
# print("Vmax de detección con aciertos:", params_vmax)

# print("----------$$$$$----------")

# print("Número de repeticiones áreas:", repetitions_a)
# print("Número de repeticones ratios detección:", repetitions_d)
# print("Número de repeticiones ratio alto ancho:", repetitions_r)
# print("Número de repeticiones salto:", repetitions_j)
# print("Número de repeticiones hmin:", repetitions_hmin)
# print("Número de repeticones smin:", repetitions_smin)
# print("Número de repeticiones vmin:", repetitions_vmin)
# print("Número de repeticiones hmax:", repetitions_hmax)
# print("Número de repeticiones smax:", repetitions_smax)
# print("Número de repeticones vmax:", repetitions_vmax)

# print("----------$$$$$----------")

# print("Mejor área:", most_frequent(list_a))
# print("Mejor ratio de detección:", most_frequent(list_d))
# print("Mejor alto ancho:", most_frequent(list_r))
# print("Mejor salto:", most_frequent(list_j))
# print("Mejor hmin:", most_frequent(list_hmin))
# print("Mejor smin:", most_frequent(list_smin))
# print("Mejor vmin:", most_frequent(list_vmin))
# print("Mejor hmax:", most_frequent(list_hmax))
# print("Mejor smax:", most_frequent(list_smax))
# print("Mejor vmax:", most_frequent(list_vmax))

print("----------$$$$$----------")

print("Número de iteraciones:",len(error))
print("Número de aciertos:", len(output))
print("Imágenes acertadas: ", true_hits)
print("Eficacia real:", np.round((len(true_hits) / len(os.listdir(f"{src}"))*100), 2),"%")
print("Eficacia en las iteraciones:", np.round(len(output)/len(error)*100, 2),"%")

end_time = datetime.now()
print('Duracion del calculo: {}'.format(end_time - start_time))

class Mail():

    def __init__(self):
        self.port = 465
        self.smtp_server_domain_name = "smtp.gmail.com"
        self.sender = "dracointhelab@gmail.com"
        self.password = ""
    
    def send(self, emails, subject, content):
        ssl_context = ssl.create_default_context()
        service = smtplib.SMTP_SSL(self.smtp_server_domain_name, self.port, context=ssl_context)
        service.login(self.sender, self.password)

        for email in emails:
            result = service.sendmail(self.sender, email, f"Subject: {subject}\n{content}")

        service.quit()

if __name__== '__main__':
    mails = input("Enter emails: ").split()
    subject = input("Enter subject: ")
    content = f"\n Numero de aciertos: {len(output)} \n\n Numero de fotos analizadas: {len(error)} \n\n Imagenes acertadas: {true_hits} \n\n Eficacia real: {eficaciaReal} \n\n Duracion: {end_time-start_time}"

    
    mail = Mail()
    mail.send(mails, subject, content)
    print("Correo enviado a %s" % mails)