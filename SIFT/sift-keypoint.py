import cv2
import numpy as np
import matplotlib.pyplot as plt


"""Se ingresan las imagenes"""
train_img = cv2.imread('train.jpg') 

query_img = cv2.imread('query.jpg')


# Show Images 
plt.figure(1)
plt.imshow(cv2.cvtColor(train_img, cv2.CV_32S))
plt.title('Imagen 1')

plt.figure(2)
plt.imshow(cv2.cvtColor(query_img, cv2.CV_32S))
plt.title('Imagen 2')


# Se convierten las imagenes a escala de grises

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

train_img_gray = to_gray(train_img)
query_img_gray = to_gray(query_img)

cv2.imwrite('imagen1_grises.jpg', train_img_gray)
cv2.imwrite('imagen2_grises.jpg', query_img_gray)

plt.figure(3)
plt.imshow(train_img_gray, cmap='gray')
plt.title('Imagen 1 en tono de grises')


plt.figure(4)
plt.imshow(query_img_gray, cmap= 'gray')
plt.title('Imagen 2 en tonos de grises')


# Se inicializa el detector de SIFT
sift = cv2.xfeatures2d.SIFT_create()
    
# Se generan los keypoints de las imagenes 
train_kp, train_desc = sift.detectAndCompute(train_img_gray, None)
query_kp, query_desc = sift.detectAndCompute(query_img_gray, None)

keypoints_img1 = (cv2.drawKeypoints(train_img_gray, train_kp, train_img.copy()))
keypoints_img2 = (cv2.drawKeypoints(query_img_gray, query_kp, query_img.copy()))

cv2.imwrite('imagen1_keypoints.jpg', keypoints_img1)
cv2.imwrite('imagen2_keypoints.jpg', keypoints_img2)

plt.figure(5)
plt.imshow((cv2.drawKeypoints(train_img_gray, train_kp, train_img.copy())))
plt.title('Imagen 1 con keypoints')

plt.figure(6)
plt.imshow((cv2.drawKeypoints(query_img_gray, query_kp, query_img.copy())))
plt.title('Imagen 2 con keypoints')


# Se identifican las coincidencias entre las imagenes
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(train_desc, query_desc)

# Se mezclan las coincidencias
matches = sorted(matches, key = lambda x:x.distance)

# Se guardan las coincidencias 
N_MATCHES = 100

match_img = cv2.drawMatches(
    train_img, train_kp,
    query_img, query_kp,
    matches[:N_MATCHES], query_img.copy(), flags=0)

cv2.imwrite('imagen_coincidencias_SIFT.jpg', match_img)
plt.figure(7)
plt.imshow(match_img)
plt.title('Deteccion SIFT')
plt.show()