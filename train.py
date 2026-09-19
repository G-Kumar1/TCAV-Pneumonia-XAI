import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle

# Data Generators
IMG_SIZE = (224,224)
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=10,

    width_shift_range=0.05,

    height_shift_range=0.05,

    zoom_range=0.05

)

val_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)



train_generator = train_datagen.flow_from_directory(

    "dataset/chest_xray_processed/train",

    target_size=(224,224),

    batch_size=32,

    class_mode="binary",

    shuffle=True

)

val_generator = val_datagen.flow_from_directory(

    "dataset/chest_xray_processed/val",

    target_size=(224,224),

    batch_size=32,

    class_mode="binary",

    shuffle=False

)




# Load model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Train the model
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(base_model.input, output)


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
model.save('models/resnet50_feature_extractor.keras')
with open("models/history.pkl","wb") as f:
    pickle.dump(history.history,f)