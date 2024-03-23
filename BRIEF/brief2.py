import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar imágenes
imagen_entrenamiento = cv2.imread('train.jpg')

# Convertir imagen a RGB
imagen_entrenamiento_rgb = cv2.cvtColor(imagen_entrenamiento, cv2.COLOR_BGR2RGB)

# Convertir la imagen de entrenamiento a escala de grises
imagen_entrenamiento_gris = cv2.cvtColor(imagen_entrenamiento_rgb, cv2.COLOR_RGB2GRAY)

# Crear imagen de prueba agregando Invarianza de Escala e Invarianza de Rotación
imagen_prueba = cv2.pyrDown(imagen_entrenamiento_rgb)
imagen_prueba = cv2.pyrDown(imagen_prueba)
num_filas, num_columnas = imagen_prueba.shape[:2]

matriz_rotacion = cv2.getRotationMatrix2D((num_columnas/2, num_filas/2), 30, 1)
imagen_prueba = cv2.warpAffine(imagen_prueba, matriz_rotacion, (num_columnas, num_filas))

imagen_prueba_gris = cv2.cvtColor(imagen_prueba, cv2.COLOR_RGB2GRAY)

# Mostrar imagen de entrenamiento y prueba
figura, subplots = plt.subplots(1, 2, figsize=(20,10))

subplots[0].set_title("Imagen de Entrenamiento")
subplots[0].imshow(imagen_entrenamiento_rgb)
#cv2.imwrite('imagen_rotada.jpg', subplots[0])

subplots[1].set_title("Imagen de Prueba")
subplots[1].imshow(imagen_prueba)

subplots[0].get_figure().savefig("Imagen_Rotada.jpg")
# ## Detectar puntos clave y crear descriptores

fast = cv2.FastFeatureDetector_create() 
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

puntos_clave_entrenamiento = fast.detect(imagen_entrenamiento_gris, None)
puntos_clave_prueba = fast.detect(imagen_prueba_gris, None)

puntos_clave_entrenamiento, descriptor_entrenamiento = brief.compute(imagen_entrenamiento_gris, puntos_clave_entrenamiento)
puntos_clave_prueba, descriptor_prueba = brief.compute(imagen_prueba_gris, puntos_clave_prueba)

puntos_clave_sin_tamaño = np.copy(imagen_entrenamiento_rgb)
puntos_clave_con_tamaño = np.copy(imagen_entrenamiento_rgb)

cv2.drawKeypoints(imagen_entrenamiento_rgb, puntos_clave_entrenamiento, puntos_clave_sin_tamaño, color=(0, 255, 0))

cv2.drawKeypoints(imagen_entrenamiento_rgb, puntos_clave_entrenamiento, puntos_clave_con_tamaño, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar imagen con y sin tamaño de puntos clave
figura, subplots = plt.subplots(1, 2, figsize=(20,10))

subplots[0].set_title("Puntos Clave de Entrenamiento Con Tamaño")
subplots[0].imshow(puntos_clave_con_tamaño, cmap='gray')

subplots[1].set_title("Puntos Clave de Entrenamiento Sin Tamaño")
subplots[1].imshow(puntos_clave_sin_tamaño, cmap='gray')

subplots[1].get_figure().savefig("Imagen_Keypoints.jpg")

# Imprimir el número de puntos clave detectados en la imagen de entrenamiento
print("Número de Puntos Clave Detectados en la Imagen de Entrenamiento: ", len(puntos_clave_entrenamiento))

# Imprimir el número de puntos clave detectados en la imagen de prueba
print("Número de Puntos Clave Detectados en la Imagen de Prueba: ", len(puntos_clave_prueba))


# ## Emparejar puntos clave

# Crear un objeto Brute Force Matcher.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Realizar el emparejamiento entre los descriptores BRIEF de la imagen de entrenamiento y la imagen de prueba
emparejamientos = bf.match(descriptor_entrenamiento, descriptor_prueba)

# Los emparejamientos con una distancia más corta son los que queremos.
emparejamientos = sorted(emparejamientos, key=lambda x: x.distance)

resultado = cv2.drawMatches(imagen_entrenamiento_rgb, puntos_clave_entrenamiento, imagen_prueba_gris, puntos_clave_prueba, emparejamientos, imagen_prueba_gris, flags=2)

# Mostrar los mejores puntos coincidentes
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Mejores Puntos Coincidentes')
plt.imshow(resultado)
plt.show()
subplots[1].get_figure().savefig("Imagen_Enlaces.jpg")

# Imprimir el número total de puntos coincidentes entre las imágenes de entrenamiento y consulta
print("\nNúmero de Puntos Clave Coincidentes Entre las Imágenes de Entrenamiento y Consulta: ", len(emparejamientos))
