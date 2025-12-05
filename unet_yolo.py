import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm #barra progresso
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from ultralytics import YOLO
from tensorflow.keras.metrics import MeanIoU, Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

os.makedirs("checkpoints", exist_ok=True)
#checkpoint epoca
checkpoint_path = "checkpoints/unet_epoch_{epoch:02d}_valLoss_{val_loss:.4f}.keras"
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=False,            
    verbose=1
)


metadata=r""  #add path
path_1 = r""  #add path
path_2 = r""  #add path
path_masks = r""  #add path

x_cache = r""  #add path
y_cache = r""  #add path

lado_cm = 1.0  #tamanho real do quadrado
pixels_por_cm = 60  #para desenhar  
lado_pixels = int(lado_cm * pixels_por_cm) 
pos_escala = (5,5) #pos na img

iou = MeanIoU(num_classes=2) #para avaliaçao de metricas
#mede a interseçao entre as mascaras sobre a uniao

debug_quad = False #testes detectar quadrado

def desenhar_quadrado_escala(img):
    img = img.copy() #copia img
    x0, y0 = pos_escala #pos na img
    x1, y1 = x0 + lado_pixels, y0 + lado_pixels
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1) #quadrado
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2) #borda
    return img

def detectar_quadrado(img_rgb, debug=False, thresh_branco = 230, min_area = 100):
    if img_rgb.dtype != np.uint8: #img em uint8 0 ou 1
        img = (img_rgb * 255).astype(np.uint8)
    else:
        img = img_rgb.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, thresh_branco, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if debug: #erro deteccao
        print("debug quadrado threshold:", thresh_branco)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: #n achou contorno
        if debug: print("nenhum contorno")
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours: #debug contorno
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if debug:
            print(f"contorno: area={area:.1f}, vertices={len(approx)}, w={w}, h={h}, aspect={aspect:.2f}")

        if len(approx) == 4 or abs(w - h) < max(6, 0.07 * max(w, h)): #achou lados quad
            escala_cm_por_pixel = lado_cm / float(w)
            if debug:
                print(f"quadrado: {w}px => {escala_cm_por_pixel:.4f} cm/pixel")
            return escala_cm_por_pixel, (x, y, w, h)
        
    if debug: #quad nao detectado
        print("nenhum quadrado detectado")
    return None
        
if os.path.exists(x_cache) and os.path.exists(y_cache): #carrega imagens da cache
    x = np.load(x_cache)
    y = np.load(y_cache)
else:
    x, y = [], []
    for folder in [path_1, path_2]: #montar cache
        for f in tqdm(os.listdir(folder)):
            if f.endswith(".jpg"):
                img_path = os.path.join(folder, f)
                mask_path = os.path.join(path_masks, f.replace(".jpg", "_segmentation.png"))

                if not os.path.exists(mask_path):
                    continue

                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (128, 128)) #imgs redimensionadas
                mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
                img = desenhar_quadrado_escala(img) #desenha na cache
                x.append(img)
                y.append(mask)#quadrado n vai na mascara para n aprender q é parte da lesao

    x = np.array(x, dtype=np.float32) / 255.0 #converte p formato numpy e pra 0.0-1.0 para usar com sigmoid
    y = np.expand_dims(np.array(y, dtype=np.float32), axis=-1) / 255.0 
                                    #formato esperado
    np.save(x_cache, x)
    np.save(y_cache, y)

#print("Arquivos em path_1:", len(os.listdir(path_1)))
#print("Arquivos em path_2:", len(os.listdir(path_2)))
#print("Arquivos em masks:", len(os.listdir(path_masks)))

#if os.path.exists(x_cache):
    #print("X cache shape:", np.load(x_cache).shape)
#if os.path.exists(y_cache):
    #print("Y cache shape:", np.load(y_cache).shape)

X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=42)
#print("Treino: ", X_train.shape, "Validacao: ", X_val.shape)

def unet_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # encoder - reduz a resolucao 
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs) #aprende os padroes
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1) #64 filtros 3x3
    p1 = layers.MaxPooling2D((2,2))(c1) #reduz res
    #aumenta filtros
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2) #para usar na skip connection

    # bottleneck - mais filtros
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b) #aqui nao reduz mais e so refina

    # decoder - aumenta a resolucao
    u2 = layers.UpSampling2D((2,2))(b) #aumenta res
    u2 = layers.Concatenate()([u2, c2]) #skip connection - une com a camada anterior correspondente
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D((2,2))(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4) #sigmoid ativa cada pixel
    #cada pixel é entre 0 e 1 e a mascara é obtida com >0.5
    return models.Model(inputs, outputs)

model = unet_model((128,128,3))
model = load_model("checkpoints/unet_epoch_30_valLoss_0.1494.keras")
model.compile(optimizer='adam', 
              loss='binary_crossentropy', #mede a diferença entre as mascaras
              metrics=['accuracy',iou,Precision(),Recall()]) #so accuracy é ruim pois pode acertar muito so com o fundo - porcentagem de pixels certos
                #iou = interseccao sobre uniao - melhor para medida
                #precision - quantos positivos sao lesao
                #recall - quantos da lesao detectados
#model.fit(X_train, Y_train, batch_size=32, epochs=30,initial_epoch=20, validation_data=(X_val, Y_val), callbacks=[checkpoint_cb]) #32 imgs antes de mudar peso
#calculo metricas

scores = model.evaluate(X_val, Y_val)
print("metricas unet")
print("loss:", scores[0])
print("accuracy:", scores[1])
print("iou:", scores[2])
print("precision:",scores[3])
print("recall:",scores[4])

#YOLO
yolo_model = YOLO(r"")  #add path
test_img_path = r""  #add path

#metricas yolo
metrics = yolo_model.val(data=r"") #add path
print("mAP50:", metrics.box.map50)

img = cv2.imread(test_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = desenhar_quadrado_escala(img_rgb)
plt.imshow(img_rgb)
plt.title("Imagem com quadrado de escala")
plt.show()

#testar yolo
results = yolo_model(img_rgb)
annotated = results[0].plot()
plt.imshow(annotated)
plt.title("deteccao yolo")
plt.axis("off")
plt.show()

quad = detectar_quadrado(img_rgb, debug=debug_quad)
if quad is None:
    print("nenhum quadrado")
    escala_cm_por_pixel = None
else:
    escala_cm_por_pixel, quadrado_bbox = quad
    print("escala detectada:", escala_cm_por_pixel, "cm/pixel")

results = yolo_model(img_rgb)[0]
#yolo detecta a lesao
#aplica unet em cada detecção do yolo e retorna a mascara
for r in results:
    for box in r.boxes: #detecçoes
        x1, y1, x2, y2 = map(int, box.xyxy[0]) #coordenadas da lesao pela caixa
        crop = img_rgb[y1:y2, x1:x2] #corta so a area da lesao
        h0, w0 = crop.shape[:2]

        crop_resized = cv2.resize(crop, (128, 128)) / 255.0 #ajusta p tamanho da unet e normaliza p 0 1
        crop_resized = np.expand_dims(crop_resized, axis=0)
        raw_pred = model.predict(crop_resized)[0,:,:,0]
        pred_mask = (raw_pred > 0.5).astype(np.uint8) #valores 0 ou 1
        #limpar ruidos
        kernel = np.ones((3,3), np.uint8)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        #print("crop_resized shape:", crop_resized.shape)
        #redimensiona a mascara p tamanho original
        mask_original = cv2.resize(pred_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
        if escala_cm_por_pixel is not None:
            area_px = int(np.sum(mask_original)) #aqui conta os pixels da lesao
            print("escala_cm_por_pixel:", escala_cm_por_pixel)
            print("quadrado_bbox:", quadrado_bbox)
            print("pixels da mascara:", np.sum(pred_mask))
            print("pixels da mascara original:", np.sum(mask_original))
            area_cm2 = area_px * (escala_cm_por_pixel ** 2) #converte
            print(f"area detectada: {area_px} px -> {area_cm2:.2f} cm2")

        plt.imshow(raw_pred, cmap='jet')
        plt.colorbar()
        plt.title("saida da Unet")
        plt.show()

        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1); plt.title("original"); plt.imshow(crop)
        plt.subplot(1,3,3); plt.title("unet segmentação"); plt.imshow(mask_original.squeeze(), cmap="gray")
        plt.show()
        print("Dim crop original:", h0, w0)
        print("Dim máscara original:", mask_original.shape)


