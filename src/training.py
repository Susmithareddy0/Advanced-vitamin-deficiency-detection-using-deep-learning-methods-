from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

train_path = "dataset/train"
val_path = "dataset/validation"

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = train_datagen.flow_from_directory(
    val_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

model = create_model()

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

model.save("vitamin_model.h5")
