import os
import numpy as np
from random import shuffle
from tensorflow.keras import layers
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
from tensorflow import keras
import efficientnet.tfkeras as efn
import swin_layers
import transformer_layers
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Layer, Dropout, Flatten
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")


data_path = 'D:/Pachigo/Covid/Medical_Data_4'


TrianImage = "D:/Pachigo/Covid/Medical_Data_4/train/"
TestImage = "D:/Pachigo/Covid/Medical_Data_4/val/"

# Normalimages = os.listdir(TrianImage + "/Normal")
# Covid_19_images = os.listdir(TrianImage + "/Covid-19")
# Pneumoniaimages = os.listdir(TrianImage + "/Pneumonia")

Bacteriaimages = os.listdir(TrianImage + "/Lung_Opacity")
Covid_19_images = os.listdir(TrianImage + "/Covid-19")
Normalimages = os.listdir(TrianImage + "/Normal")
Virusimages = os.listdir(TrianImage + "/Viral_Pneumonia")


Nos_Train = len(Bacteriaimages) + len(Covid_19_images) + len(Normalimages) + len(Virusimages)
# Nos_Train = len(Covid_19_images) + len(Normalimages) + len(Pneumoniaimages)
# Nos_Train = len(Normalimages) + len(Pneumoniaimages)


image_size = 224
BATCH_SIZE = 8
STEPS_PER_EPOCH = int(Nos_Train // BATCH_SIZE)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   zoom_range=0.2,
                                   rotation_range=15,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(data_path + '/train',
                                                 target_size=(image_size, image_size),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical', shuffle=True)

testing_set = test_datagen.flow_from_directory(data_path + '/val',
                                               target_size=(image_size, image_size),
                                               batch_size=BATCH_SIZE,
                                               class_mode='categorical', shuffle=False)


def display_training_curves(training_accuracy, validation_accuracy, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_accuracy)
    ax.plot(validation_accuracy)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Accuracy', 'Val_Accuracy'])


def display_training_curves2(training_loss, validation_loss, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_loss)
    ax.plot(validation_loss)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Loss', 'Val_Loss'])


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # map the activations of the last conv layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, top_pred_index.numpy()


def superimposed_img(image, heatmap):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image_size, image_size))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + image
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


# label smoothing
def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


# training call backs
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, epsilon=0.00001, patience=10,
                                                 verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
counter = Counter(training_set.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


# Lets build our Ensemble network -- this is the pretrained model
pretrained_D = tf.keras.applications.DenseNet201(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                 include_top=False)
# pretrained_E = efn.EfficientNetB7(input_shape=(image_size, image_size, 3), weights='noisy-student', include_top=False)
# pretrained_VGG = tf.keras.applications.VGG16(input_shape=(image_size, image_size, 3), weights='imagenet',
#                                             include_top=False)
# pretrained_googleNet = tf.keras.applications.InceptionV3(input_shape=(image_size, image_size, 3), weights='imagenet',
#                                                         include_top=False)
pretrained_IRV2 = tf.keras.applications.InceptionResNetV2(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                          include_top=False)
pretrained_Xception = tf.keras.applications.Xception(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                     include_top=False)

# Combining the models
for layer in pretrained_D.layers:
    layer.trainable = False

for layer in pretrained_IRV2.layers:
    layer.trainable = False

for layer in pretrained_Xception.layers:
    layer.trainable = False

# for layer in pretrained_E.layers:
#     layer.trainable = False

# for layer in pretrained_VGG.layers:
#     layer.trainable = False

# for layer in pretrained_googleNet.layers:
#     layer.trainable = False

visible = tf.keras.layers.Input(shape=(image_size, image_size, 3))

x1 = pretrained_D(visible)
x1 = tf.keras.layers.AveragePooling2D()(x1)
x1 = tf.keras.layers.Flatten()(x1)
x1 = tf.keras.layers.Dense(32, activation="relu")(x1)
x1 = tf.keras.layers.Dropout(0.5)(x1)

# x3 = pretrained_VGG(visible)
# x3 = tf.keras.layers.AveragePooling2D()(x3)
# x3 = tf.keras.layers.Flatten()(x3)
# x3 = tf.keras.layers.Dense(32, activation="relu")(x3)
# x3 = tf.keras.layers.Dropout(0.5)(x3)

# x2 = pretrained_googleNet(visible)
# x2 = tf.keras.layers.ZeroPadding2D(padding=((0, 2), (0, 2)))(x2)
# x2 = tf.keras.layers.AveragePooling2D()(x2)
# x2 = tf.keras.layers.Flatten()(x2)
# x2 = tf.keras.layers.Dense(32, activation="relu")(x2)
# x2 = tf.keras.layers.Dropout(0.5)(x2)

x4 = pretrained_IRV2(visible)
x4 = tf.keras.layers.ZeroPadding2D(padding=((0, 2), (0, 2)))(x4)
x4 = tf.keras.layers.AveragePooling2D()(x4)
x4 = tf.keras.layers.Flatten()(x4)
x4 = tf.keras.layers.Dense(32, activation="relu")(x4)
x4 = tf.keras.layers.Dropout(0.5)(x4)

x5 = pretrained_Xception(visible)
x5 = tf.keras.layers.AveragePooling2D()(x5)
x5 = tf.keras.layers.Flatten()(x5)
x5 = tf.keras.layers.Dense(32, activation="relu")(x5)
x5 = tf.keras.layers.Dropout(0.5)(x5)

merge = tf.keras.layers.concatenate([x1, x4, x5], name="concatallprobs")
x = tf.keras.layers.Dense(32, activation='relu')(merge)
x = tf.keras.layers.Dropout(0.5)(x)
OUT = tf.keras.layers.Dense(4, activation='softmax')(x)

model = Model(inputs=visible, outputs=OUT)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.2), loss=categorical_smooth_loss,
              metrics=['accuracy'])
# model.summary()

if __name__ == "__main__":
    #history = model.fit(training_set, validation_data=testing_set, epochs=50)  # 30
    history = model.fit(training_set, validation_data=testing_set, callbacks=[lr_reduce, es_callback], epochs=100)

    display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    plt.show()

    # Saving the Model
    model.save('Ensemble4.h5')
