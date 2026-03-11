import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

model = load_model("maize_disease_detection_new_model.h5")

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    "Traning/Dataset/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)

print(classification_report(test_generator.classes, y_pred))