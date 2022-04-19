import tensorflow.keras.losses
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pycm import *
import ROC_AUC
import itertools
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformer_layers import patch_extract, patch_embedding, patch_merging
from swin_layers import SwinTransformerBlock, Mlp, WindowAttention

def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss

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


data_path = 'D:/Pachigo/Covid/Medical_Data_3'

keras.losses.categorical_smooth_loss = categorical_smooth_loss
reconstructed_model = keras.models.load_model("model_transformer_3.h5",custom_objects={'patch_extract': patch_extract,'patch_embedding':patch_embedding,'Mlp':Mlp,
                                                                                       'WindowAttention':WindowAttention,'SwinTransformerBlock':SwinTransformerBlock,'patch_merging':patch_merging,'categorical_smooth_loss':categorical_smooth_loss})
#reconstructed_model.summary()
image_size = 224
BATCH_SIZE = 4
test_datagen = ImageDataGenerator(rescale=1. / 255)

testing_set = test_datagen.flow_from_directory(data_path + '/val',
                                               target_size=(image_size, image_size),
                                               batch_size=1,
                                               class_mode='categorical', shuffle=False)

# target_names = ['Covid-19', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
target_names = ['Covid-19', 'Normal', 'Pneumonia']
# target_names = ['Normal', 'Pneumonia']


y_pred, y_true = list(), list()

total_samples = testing_set.n
counter_q = 0
# for i, b in enumerate(testing_set)
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


# last_conv_layer_name = "patch_merging"
# classifier_layer_names = [
#     "Globa",
#     "Dense"
# ]
#
# labels = ['Normal', 'Covid-19', 'Pneumonia']

# file_path = "G:/Covid/Medical_Data_3/val/Normal/Normal(1).jpg"
# test_image = cv2.imread(file_path)
# test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
# plt.imshow(test_image)

# test_image = np.expand_dims(test_image,axis=0)
# heatmap, top_index = make_gradcam_heatmap(test_image, reconstructed_model, last_conv_layer_name, classifier_layer_names)

# print("predicted as", labels[top_index])
# plt.matshow(heatmap)
# plt.show()

# s_img = superimposed_img(test_image[0], heatmap)
# plt.imshow(s_img)

# sample_data = testing_set.__getitem__(0)[0]
# sample_label = testing_set.__getitem__(0)[1]
#
# plt.figure(figsize=(10, 8))
# for i in range(12):
#     plt.subplot(3, 4, i + 1)
#     plt.axis('off')
#     heatmap, top_index = make_gradcam_heatmap(np.expand_dims(sample_data[i], axis=0), reconstructed_model,
#                                               last_conv_layer_name, classifier_layer_names)
#     img = np.uint8(255 * sample_data[i])
#     s_img = superimposed_img(img, heatmap)
#     plt.imshow(s_img)
#     plt.title(labels[np.argmax(sample_label[i])] + " pred as: " + labels[top_index], fontsize=8)
