import os
import numpy as np
# import cv2
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
BATCH_SIZE = 4
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
#                                                          include_top=False)
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

# x4 = pretrained_E(visible)
# x3 = pretrained_VGG(visible)
# x2 = pretrained_googleNet(visible)

x1 = pretrained_D(visible)
x5 = pretrained_IRV2(visible)
x6 = pretrained_Xception(visible)

# x2 = tf.keras.layers.ZeroPadding2D(padding=((0, 2), (0, 2)))(x2)
x5 = tf.keras.layers.ZeroPadding2D(padding=((0, 2), (0, 2)))(x5)
merge = tf.keras.layers.concatenate([x1, x5, x6], name="concatallprobs")
part_A = tf.keras.layers.ZeroPadding2D(padding=((0, 5), (0, 5)))(merge)

# input_size = (8, 8, 4992)  # model input
# input_size = (12, 12, 4480)
input_size = (12, 12, 5504)
patch_size = (2, 2)  # Segment 28-by-28 frames into 2-by-2 sized patches, patch contents and positions are embedded
n_labels = 4  # Data_Classes

# Dropout parameters
mlp_drop_rate = 0.01  # Droupout after each MLP layer
attn_drop_rate = 0.01  # Dropout after Swin-Attention
proj_drop_rate = 0.01  # Dropout at the end of each Swin-Attention block, i.e., after linear projections
drop_path_rate = 0.01  # Drop-path within skip-connections

num_heads = 8  # Number of attention heads 8
embed_dim = 64  # Number of embedded dimensions 64
num_mlp = 256  # Number of MLP nodes 256
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
qk_scale = None  # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
window_size = 2  # Size of attention window (height = width)
shift_size = window_size // 2  # Size of shifting (shift_size < window_size)

num_patch_x = input_size[0] // patch_size[0]
num_patch_y = input_size[1] // patch_size[1]

patch_size = patch_size[0]
X = transformer_layers.patch_extract(patch_size)(part_A)
X = transformer_layers.patch_embedding(num_patch_x * num_patch_y, embed_dim)(X)

for i in range(2):
    if i % 2 == 0:
        shift_size_temp = 0
    else:
        shift_size_temp = shift_size

    X = swin_layers.SwinTransformerBlock(dim=embed_dim, num_patch=(num_patch_x, num_patch_y), num_heads=num_heads,
                                         window_size=window_size, shift_size=shift_size_temp, num_mlp=num_mlp,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         mlp_drop=mlp_drop_rate, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate,
                                         drop_path_prob=drop_path_rate,
                                         prefix='swin_block{}'.format(i))(X)

X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
X = GlobalAveragePooling1D()(X)
X = tf.keras.layers.Dense(64, activation="relu", name="dense_head1")(X)
X = tf.keras.layers.Dropout(0.5, name="dropout_head1")(X)
OUT = Dense(n_labels, activation='softmax')(X)

model = Model(inputs=visible, outputs=OUT)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.2), loss=categorical_smooth_loss,
              metrics=['accuracy'])
# model.summary()


if __name__ == "__main__":
    # history = model.fit(training_set, validation_data=testing_set, epochs=100)  # 30
    history = model.fit(training_set, validation_data=testing_set, callbacks=[lr_reduce, es_callback], epochs=100)

    display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
    display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
    plt.show()

    # Saving the Model
    model.save('model_transformerB_4.h5')
    model.save_weights("PachigoB_4")
