import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

# Check GPU Availability
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU being used:", tf.test.gpu_device_name())

# Enable Memory Growth (Prevents GPU Overload)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth enabled for GPUs")

# Enable Mixed Precision (Boosts Speed with Tensor Cores)
mixed_precision.set_global_policy('mixed_float16')

# Define paths to dataset
dataset_path = "datasets/train"  # Update to the correct path

# Image dimensions
img_height, img_width = 224, 224
batch_size = 32  # Increased for better GPU utilization

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,   
    rotation_range=25,    
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    validation_split=0.2  
)

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Load Pretrained Xception Model
base_model = tf.keras.applications.Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Allow fine-tuning
base_model.trainable = True

# Define Model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")  
])

# Compile the model with mixed precision optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.000005)
opt = mixed_precision.LossScaleOptimizer(opt)

model.compile(
    optimizer=opt,  
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model with Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#Explicitly Run Training on GPU
with tf.device('/GPU:0'):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,  # Reduced epochs for stability
        callbacks=[early_stopping],
        verbose=1
    )

# Save the trained model
model.save("models/new_deepfake_detector.h5")  
print("Model training complete! Saved at models/new_deepfake_detector.h5")
