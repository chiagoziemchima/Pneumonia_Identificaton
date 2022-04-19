import keras.losses
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pycm import *
import ROC_AUC


def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss

keras.losses.categorical_smooth_loss = categorical_smooth_loss

data_path = 'D:/Pachigo/Covid/Medical_Data_3'

reconstructed_model = keras.models.load_model("model_E.h5")

image_size = 224
BATCH_SIZE = 4
test_datagen = ImageDataGenerator(rescale=1. / 255)

testing_set = test_datagen.flow_from_directory(data_path + '/val',
                                               target_size=(image_size, image_size),
                                               batch_size=1,
                                               class_mode='categorical', shuffle=False)

# arget_names = ['Covid-19', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
target_names = ['Covid-19', 'Normal', 'Pneumonia']
# target_names = ['Covid-19', 'Pneumonia']

y_pred, y_true = list(), list()

total_samples = testing_set.n
counter_q = 0
#for i, b in enumerate(testing_set)
while True :
    b = next(testing_set)
    shapes = b[0].shape
    counter_q = counter_q+shapes[0]
    y_p = reconstructed_model.predict(b[0])
    y_pred.append(np.argmax(y_p, axis=1)[0])
    y_true.append(np.argmax(b[1], axis=1)[0])
    if counter_q == total_samples:
        break

prec_rec = classification_report(y_true, y_pred, target_names=target_names)
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
cm_b = ConfusionMatrix(y_true, y_pred)
print(f"Accuracy : {accuracy}")
with open("prec_rec.txt", "a") as f:
    print(prec_rec, file=f)
with open("cm.txt", "a") as f:
    print(cm, file=f)
with open("cm_ext.txt", "a") as f:
    print(cm_b, file=f)
ROC_AUC.plot_ROC(y_true, y_pred, classes=[0, 1, 2])  # 2
ROC_AUC.plot_PrecRec(y_true, y_pred, classes=[0, 1, 2])  # 2