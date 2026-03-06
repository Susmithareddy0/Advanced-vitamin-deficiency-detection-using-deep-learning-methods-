import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("vitamin_model.h5")

classes = ['Vitamin A Deficiency', 'Vitamin B12 Deficiency', 'Vitamin D Deficiency', 'Healthy']

img = image.load_img("test_image.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

result = classes[np.argmax(prediction)]

print("Prediction:", result)
