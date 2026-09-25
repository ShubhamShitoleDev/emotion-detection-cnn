#################################################################################
#
#  Project Name  : Real-Time Emotion Detection (CNN) - Confusion Matrix Analysis
#  Description   : Loads the trained emotion CNN, runs predictions on the full
#                   test set, and builds a confusion matrix showing exactly
#                   which emotion classes get confused with each other -
#                   quantifies the Happy/Surprise confusion observed during
#                   live testing, and reveals any other class overlaps.
#  Date          : 24-Sep-2026
#  Author        : Shubham Shitole
#
#################################################################################

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Border = "-" * 50

MODEL_PATH = "emotion_cnn_model.keras"
TEST_DIR = r"C:\Users\shubh\Desktop\Deep_Learning\emotion-detection-cnn\data\test"
EMOTION_CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48
BATCH_SIZE = 64


#################################################################################
#
# Function Name : load_test_generator
# Input :         test_dir
# Description :   Builds a test data generator (no augmentation, no shuffle -
#                 order must stay fixed so predictions can be matched back
#                 to true labels correctly)
# Return Value :  test_generator
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_test_generator(test_dir):
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        classes=EMOTION_CLASSES,
        class_mode='sparse',
        shuffle=False
    )

    return test_generator


#################################################################################
#
# Function Name : get_predictions
# Input :         model, test_generator
# Description :   Runs the model on every image in the test set and collects
#                 both the predicted and true labels for comparison
# Return Value :  y_true (array of true labels), y_pred (array of predicted labels)
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def get_predictions(model, test_generator):
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    return y_true, y_pred


#################################################################################
#
# Function Name : plot_confusion_matrix
# Input :         y_true, y_pred, class_names
# Description :   Builds a confusion matrix and saves it as a heatmap image -
#                 rows are the true emotion, columns are the predicted
#                 emotion, so off-diagonal cells show exactly which classes
#                 the model confuses with each other
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("True Emotion")
    ax.set_title("Emotion Detection - Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved : confusion_matrix.png")


#################################################################################
#
# Function Name : main
# Description :   Loads the model and test data, generates predictions,
#                 prints a per-class classification report, and saves the
#                 confusion matrix visualization
# Date :          24-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load Model and Test Data")
    print(Border)

    model = tf.keras.models.load_model(MODEL_PATH)
    test_generator = load_test_generator(TEST_DIR)

    print(Border)
    print("Step 2 : Generate Predictions on Full Test Set")
    print(Border)

    y_true, y_pred = get_predictions(model, test_generator)

    print(Border)
    print("Step 3 : Classification Report")
    print(Border)

    print(classification_report(y_true, y_pred, target_names=EMOTION_CLASSES))

    print(Border)
    print("Step 4 : Build Confusion Matrix")
    print(Border)

    plot_confusion_matrix(y_true, y_pred, EMOTION_CLASSES)


if __name__ == "__main__":
    main()