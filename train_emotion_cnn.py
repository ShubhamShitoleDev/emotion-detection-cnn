#################################################################################
#
#  Project Name  : Real-Time Emotion Detection (CNN)
#  Description   : Trains a CNN on the FER2013 facial expression dataset
#                   (folder-structured: one subfolder per emotion) to classify
#                   6 emotions - angry, fear, happy, neutral, sad, surprise
#                   (the 'disgust' class is excluded due to very few samples).
#                   Uses data augmentation to help generalize on this
#                   relatively small, noisy dataset.
#  Date          : 24-Sep-2026
#  Author        : Shubham Shitole
#
#################################################################################

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Border = "-" * 50

TRAIN_DIR = r"C:\Users\shubh\Desktop\Deep_Learning\emotion-detection-cnn\data\train"
TEST_DIR = r"C:\Users\shubh\Desktop\Deep_Learning\emotion-detection-cnn\data\test"

EMOTION_CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 30


#################################################################################
#
# Function Name : build_data_generators
# Input :         train_dir, test_dir
# Description :   Builds ImageDataGenerators for train (with augmentation -
#                 rotation, shift, zoom, flip - to help the model generalize
#                 on this small dataset) and test (no augmentation, just
#                 normalization). flow_from_directory automatically infers
#                 class labels from subfolder names, and the 'classes'
#                 parameter restricts loading to only our 6 chosen emotions,
#                 excluding 'disgust'.
# Return Value :  train_generator, test_generator
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_data_generators(train_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        classes=EMOTION_CLASSES,
        class_mode='sparse'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        classes=EMOTION_CLASSES,
        class_mode='sparse',
        shuffle=False
    )

    print("Train samples : ", train_generator.samples)
    print("Test samples : ", test_generator.samples)
    print("Class indices : ", train_generator.class_indices)

    return train_generator, test_generator


#################################################################################
#
# Function Name : build_cnn_model
# Input :         input_shape, num_classes
# Description :   Builds a 4-block CNN for emotion classification - deeper
#                 than the CIFAR-10 model since facial expressions involve
#                 subtler visual differences than distinct object categories
# Return Value :  Compiled-ready Keras Sequential model
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_cnn_model(input_shape=(48, 48, 1), num_classes=6):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model


#################################################################################
#
# Function Name : train_model
# Input :         model, train_generator, test_generator
# Description :   Compiles and trains the model directly on the generators
#                 (which yield augmented batches on the fly), using
#                 EarlyStopping to restore the best-performing weights
# Return Value :  history object from model.fit
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def train_model(model, train_generator, test_generator):
    print(Border)
    print("Training : Emotion Detection CNN")
    print(Border)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    return history


#################################################################################
#
# Function Name : main
# Description :   Runs the full pipeline: build data generators, build model,
#                 train, evaluate, and save the model for use in the
#                 real-time webcam emotion detection script
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Build Data Generators")
    print(Border)

    train_generator, test_generator = build_data_generators(TRAIN_DIR, TEST_DIR)

    print(Border)
    print("Step 2 : Build CNN Model")
    print(Border)

    model = build_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(EMOTION_CLASSES))
    model.summary()

    print(Border)
    print("Step 3 : Train Model")
    print(Border)

    history = train_model(model, train_generator, test_generator)

    print(Border)
    print("Step 4 : Final Results")
    print(Border)

    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print(f"Final Test Accuracy (best restored model) : {test_acc*100:.2f}%")

    model.save("emotion_cnn_model.keras")
    print("Model saved as emotion_cnn_model.keras")


if __name__ == "__main__":
    main()
