import os
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
from tensorflow import keras
import efficientnet.tfkeras as efn


data_path = 'C:/Users/25715/Desktop/Covid/Medical_Data_3'


TrianImage = "C:/Users/25715/Desktop/Covid/Medical_Data_3/train/"
TestImage = "C:/Users/25715/Desktop/Covid/Medical_Data_3/val/"


Normalimages = os.listdir(TrianImage + "/Normal")
Covid_19_images = os.listdir(TrianImage + "/Covid-19")
Pneumoniaimages = os.listdir(TrianImage + "/Pneumonia")


Nos_Train = len(Covid_19_images) + len(Normalimages) + len(Pneumoniaimages)

image_size = 224
BATCH_SIZE = 4
STEPS_PER_EPOCH = int(Nos_Train // BATCH_SIZE)

# plt.figure(figsize=(10,10))
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(plt.imread(os.path.join(TrianImage + "/Bacteria", Bacteriaimages[i])),cmap='gray')
#     plt.imshow(plt.imread(os.path.join(TrianImage + "/Covid-19", Covid_19_images[i])), cmap='gray')
#     plt.imshow(plt.imread(os.path.join(TrianImage + "/Normal", Normalimages[i])), cmap='gray')
#     plt.imshow(plt.imread(os.path.join(TrianImage + "/Virus", Virusimages[i])), cmap='gray')
#     plt.title("Bacteria")
#     plt.title("Covid-19")
#     plt.title("Normal")
#     plt.title("Virus")
# plt.show()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   rotation_range=15,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(data_path + '/train',
                                                 target_size = (image_size, image_size),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical', shuffle=True)

testing_set = test_datagen.flow_from_directory(data_path + '/val',
                                               target_size = (image_size, image_size),
                                               batch_size = BATCH_SIZE,
                                               class_mode = 'categorical', shuffle = True)


# print(training_set.class_indices)
# print(testing_set.class_indices)
#
# labels = ['Bacteria', 'Covid-19', 'Normal', 'Virus']
# sample_data = testing_set.__getitem__(1)[0]
# sample_label = testing_set.__getitem__(1)[1]
#
# plt.figure(figsize=(10,8))
# for i in range(12):
#     plt.subplot(3, 4, i + 1)
#     plt.axis('off')
#     plt.imshow(sample_data[i])
#     plt.title(labels[np.argmax(sample_label[i])])
# plt.show()


def display_training_curves(training_accuracy, validation_accuracy, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_accuracy)
    ax.plot(validation_accuracy)
    ax.set_title('Model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Accuracy', 'Val_Accuracy'])


def display_training_curves2(training_loss, validation_loss, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_loss)
    ax.plot(validation_loss)
    ax.set_title('Model '+ title)
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
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.0001, patience=3, verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
counter = Counter(training_set.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

# print(class_weights)
# print(tf.keras.applications.DenseNet201(weights='imagenet').input_shape)

pretrained_efnet = efn.EfficientNetB7(input_shape=(image_size, image_size, 3), weights='noisy-student', include_top=False)

for layer in pretrained_efnet.layers:
  layer.trainable = False

x2 = pretrained_efnet.output
x2 = tf.keras.layers.AveragePooling2D(name="averagepooling2d_head")(x2)
x2 = tf.keras.layers.Flatten(name="flatten_head")(x2)
x2 = tf.keras.layers.Dense(64, activation="relu", name="dense_head")(x2)
x2 = tf.keras.layers.Dropout(0.5, name="dropout_head")(x2)
model_out = tf.keras.layers.Dense(3, activation='softmax', name="predictions_head")(x2)

model_E = Model(inputs=pretrained_efnet.input, outputs=model_out)
model_E.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=categorical_smooth_loss,metrics=['accuracy'])
#model_E.summary()

if __name__ == "__main__":
    history = model_E.fit(training_set, validation_data=testing_set, epochs=50) #30

    display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    plt.show()

    # Saving the Model
    model_E.save("model_E.h5")