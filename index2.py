#!/usr/bin/env python
# coding: utf-8

# 1. Install Dependencies and Setup
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

# Set GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Check GPU availability
print("Available GPUs:", tf.config.list_physical_devices('GPU'))


# 2. Remove dodgy images
data_dir = 'data'  # Directory containing the images
image_exts = ['jpeg', 'jpg', 'bmp', 'png']  # Accepted image extensions

for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            file_type = imghdr.what(image_path)
            if file_type not in image_exts: 
                print(f"Removing invalid image: {image_path}")
                os.remove(image_path)
        except Exception as e: 
            print(f"Issue with image {image_path}: {e}")
            # Optionally, remove the problematic image
            # os.remove(image_path)


# 3. Load Data
data = tf.keras.utils.image_dataset_from_directory('data')  # Load dataset

# Create a data iterator for batch processing
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Display first 4 images in the batch
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(f"Class: {batch[1][idx]}")

# 4. Scale Data and Apply One-Hot Encoding
data = data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=3)))  # Normalize images and one-hot encode labels

# Check the result
print(data.as_numpy_iterator().next())


# 5. Split Data into Train, Validation, and Test sets
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


# 6. Build Deep Learning Model (Softmax için güncelleme)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 sınıf için softmax (cat, dog, bird)
])

# Compile the model (loss fonksiyonunu 'categorical_crossentropy' yapıyoruz)
model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# Display model summary
model.summary()


# 7. Train the Model
logdir = 'logs'  # Log directory for TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# 8. Plot Performance (Loss and Accuracy)
# Plot loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Plot accuracy
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# 9. Evaluate Model on Test Set
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

# Update metrics for each batch in the test set
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

# Print evaluation results
print(f"Precision: {pre.result().numpy()}")
print(f"Recall: {re.result().numpy()}")
print(f"Accuracy: {acc.result().numpy()}")


# 10. Test with a New Image (Softmax'e göre güncelleme)
img = cv2.imread('cat.4003.jpg')

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
    yhat = model.predict(np.expand_dims(resize / 255, 0))

    # Class names
    class_names = ['cat', 'dog', 'bird']  # Yeni sınıflar

    # Display the predicted class using argmax
    predicted_class = class_names[np.argmax(yhat)]  # En yüksek olasılığa sahip sınıfı seç
    print(f'Predicted class is {predicted_class}')


# 11. Save the Model
model.save(os.path.join('models', 'imageclassifier.h5'))  # Save model

# Load the saved model
new_model = tf.keras.models.load_model('models/imageclassifier.h5')

# Test prediction with the loaded model
yhat = new_model.predict(np.expand_dims(resize / 255, 0))
predicted_class = class_names[np.argmax(yhat)]  # En yüksek olasılığa sahip sınıfı seç
print(f'Predicted class is {predicted_class}')
