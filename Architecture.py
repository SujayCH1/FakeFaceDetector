import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#  GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")

# GPU checking
if tf.config.list_physical_devices('GPU'):
    print("GPU is available. Using GPU for training.")
    device = "/gpu:0"
else:
    print("GPU is not available. Using CPU for training.")
    device = "/cpu:0"

def create_model(img_height, img_width, fine_tune_at=100):
    with tf.device(device):

        base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                                 include_top=False,
                                 weights='imagenet')
        
        base_model.trainable = False

        for layer in base_model.layers[-fine_tune_at:]:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)

        # Construct the full model
        model = Model(inputs=base_model.input, outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    return model

def save_full_model(model, save_path):
    model.save(save_path)
    print(f"Full model saved to {save_path}")

def train_model(model, train_generator, validation_generator, epochs):
    with tf.device(device):
        # train on the laters at haed
        history1 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=5,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size
        )

        # improvising
        model.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history2 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs-5,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size
        )
    
    history = {}
    history['accuracy'] = history1.history['accuracy'] + history2.history['accuracy']
    history['val_accuracy'] = history1.history['val_accuracy'] + history2.history['val_accuracy']
    history['loss'] = history1.history['loss'] + history2.history['loss']
    history['val_loss'] = history1.history['val_loss'] + history2.history['val_loss']
    
    return history

def setup_data_generators(data_dir, img_height, img_width, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7,1.3],
        channel_shift_range=50,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def plot_training_history(history, save_dir):

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')


    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

    print(f"Training history plot saved to {os.path.join(save_dir, 'training_history.png')}")

if __name__ == "__main__":
    
    img_height, img_width = 224, 224
    batch_size = 32
    epochs = 20
    data_dir = 'C:\\Users\\sujun\\Documents\\Projects\\PythonCV\\AntiSpoofing\\DATA\\Preprocessed'
    model_save_path = 'C:\\Users\\sujun\\Documents\\Projects\\PythonCV\\AntiSpoofing\\DATA\\MODEL\\anti_spoofing_model.h5'
    save_dir = 'C:\\Users\\sujun\\Documents\\Projects\\PythonCV\\AntiSpoofing\\DATA\\PLOTS'

    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    
    train_generator, validation_generator = setup_data_generators(data_dir, img_height, img_width, batch_size)

    
    model = create_model(img_height, img_width)
    model.summary()
    history = train_model(model, train_generator, validation_generator, epochs)
    plot_training_history(history, save_dir)

    
    save_full_model(model, model_save_path)

    print("Model training completed and full model saved.")