
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt



# Load the saved model
new_model = tf.keras.models.load_model('models/imageclassifier.h5')
# 10. Test with a New Image (Softmax'e göre güncelleme)
img = cv2.imread('bird.png')

if img is None:
    print("Görüntü dosyası okunamadı.")
else:
    print(f"Görüntü başarıyla okundu. Boyut: {img.shape}")
    # Convert the image to RGB (if it’s in BGR format from OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image for prediction
    resize = tf.image.resize(img, (256, 256))

    # Display the resized image
    plt.imshow(resize.numpy().astype(int))
    plt.show()

    # Make prediction
    yhat = new_model.predict(np.expand_dims(resize / 255, 0))

    # Class names
    class_names = ['bird', 'cat', 'dog']  # Yeni sınıflar

    # Display the predicted class using argmax
    predicted_class = class_names[np.argmax(yhat)]  # En yüksek olasılığa sahip sınıfı seç
    print(f'Predicted class is {predicted_class}')





# Test prediction with the loaded model
yhat = new_model.predict(np.expand_dims(resize / 255, 0))
predicted_class = class_names[np.argmax(yhat)]  # En yüksek olasılığa sahip sınıfı seç
print(f'Predicted class is {predicted_class}')
