import tensorflow as tf
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


IMG_SIZE = (224,224)
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=10,

    width_shift_range=0.05,

    height_shift_range=0.05,

    zoom_range=0.05

)
train_generator = train_datagen.flow_from_directory(

    "dataset/chest_xray_processed/train",

    target_size=(224,224),

    batch_size=32,

    class_mode="binary",

    shuffle=True

)
val_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

val_generator = val_datagen.flow_from_directory(

    "dataset/chest_xray_processed/val",

    target_size=(224,224),

    batch_size=32,

    class_mode="binary",

    shuffle=False

)


# Load previously trained feature extractor model
model = load_model("models/resnet50_feature_extractor.keras")
# Unfreeze everything
for layer in model.layers:
    layer.trainable = True

# Freeze all except last 30 layers
for layer in model.layers[:-30]:
    layer.trainable = False

# Compile with a small learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Fine-tune
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)

# Save fine-tuned model
model.save("models/resnet50_fine_tuned.keras")

# Save fine-tuning history
with open("models/resnet50_fine_tuned_history.pkl", "wb") as f:
    pickle.dump(history_fine.history, f)

