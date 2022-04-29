import os
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers

from sklearn import metrics

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2

# The path to the directory of the program dataset
# base_dir = "/home/m_hamshi/Dataset/dataset50/"
base_dir = "/home/m_hamshi/Dataset/NewArrange/dataset50-2/"

current_file_name = Path(__file__).stem

# Directories for training, validation and test splits
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test2')

# conv_base = models.load_model("/home/m_hamshi/models/InceptionResNetV2_conv_base_128.h5")

conv_base = InceptionResNetV2(weights=None, include_top=False,
                              input_shape=(128, 128, 3))

"""
def save_log_csv(file_name):
    # path = os.path.join("./results/", f"{file_name}.csv")
    path = os.path.join("/home/m_hamshi/results/", f"{file_name}.csv")
    history_logger = tf.keras.callbacks.CSVLogger(
        path, separator=",", append=True)
    return history_logger


history_logger = save_log_csv(current_file_name)
"""

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.7))
model.add(layers.Dense(1, activation='sigmoid'))

model.load_weights('/home/m_hamshi/models/weights_IRV2_imagenet.h5')

conv_base.trainable = True


def preprocess_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=(0.8, 1.2))

    # All images will be rescaled by 1./255
    # train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(128, 128),
        color_mode="rgb",
        batch_size=50,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        color_mode="rgb",
        batch_size=50,
        class_mode='binary',
        shuffle=False)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        color_mode="rgb",
        batch_size=50,
        class_mode='binary',
        shuffle=False)

    return train_generator, validation_generator, test_generator


train_generator, validation_generator, test_generator = preprocess_data()

"""
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=3e-6),
              # optimizer=optimizers.RMSprop(lr=2e-3),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    callbacks=[history_logger],
    validation_data=validation_generator,
    validation_steps=20
)

# conv_base.trainable = True

# set_trainable = False

# for layer in conv_base.layers:
#     if layer.name == 'block8_10_conv':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False


# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.Adam(learning_rate=1e-5),
#               # optimizer=optimizers.RMSprop(lr=2e-3),
#               metrics=['acc'])


# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     callbacks=[history_logger],
#     validation_data=validation_generator,
#     validation_steps=20
# )
""" 

predictions = model.predict_generator(validation_generator, steps=20)
val_preds = predictions
val_trues = validation_generator.classes

predictions = model.predict_generator(test_generator, steps=80)
test_preds = predictions
test_trues = test_generator.classes

# model.save_weights("/home/m_hamshi/models/weights_IRV2_imagenet.h5")

def plot_learningCurve(history, epochs):
    """
    Plots training & validation accuracy and loss scores

    -> Args:
    """

    # Plots training & validation accuracy scores
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['acc'])
    plt.plot(epoch_range, history.history["val_acc"])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    # plt.savefig('./results/Accuracy.png')
    # plt.savefig('/content/drive/MyDrive/Accuracy.png')
    plt.savefig(f'/home/m_hamshi/results/{current_file_name}_Accuracy.png')
    plt.close()
    # plt.show()

    # Plots training & validation loss scores
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    # plt.savefig('./results/Loss.png')
    # plt.savefig('/content/drive/MyDrive/Loss.png')
    plt.savefig(f'/home/m_hamshi/results/{current_file_name}_Loss.png')
    plt.close()
    # plt.show()

def plot_roccurve():
    file_name = current_file_name 

    fpr_val , tpr_val , thresholds_val = metrics.roc_curve(val_trues, val_preds)
    fpr_test , tpr_test , thresholds_test = metrics.roc_curve(test_trues, test_preds)

    dataframe1 = pd.DataFrame({'val_trues': np.ravel(val_trues), 'val_preds': np.ravel(val_preds)})
    dataframe2 = pd.DataFrame({'fpr_val': np.ravel(fpr_val), 'tpr_val': np.ravel(tpr_val)})
    csv_path1 = os.path.join("/home/m_hamshi/results/", f"{file_name}_preds_val.csv")
    csv_path2 = os.path.join("/home/m_hamshi/results/", f"{file_name}_fpr_val.csv")
    dataframe1.to_csv(csv_path1) 
    dataframe2.to_csv(csv_path2) 

    dataframe3 = pd.DataFrame({'test_trues': np.ravel(test_trues), 'test_preds': np.ravel(test_preds)})
    dataframe4 = pd.DataFrame({'fpr_test': np.ravel(fpr_test), 'tpr_test': np.ravel(tpr_test)})
    csv_path3 = os.path.join("/home/m_hamshi/results/", f"{file_name}_preds_test.csv")
    csv_path4 = os.path.join("/home/m_hamshi/results/", f"{file_name}_fpr_test.csv")
    dataframe3.to_csv(csv_path3) 
    dataframe4.to_csv(csv_path4) 

    plt.plot(fpr_val, tpr_val) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    path_val = os.path.join("/home/m_hamshi/results/", f"{file_name}_ROC_val.png")
    plt.savefig(path_val)
    plt.close() 
    # plt.show() 

    plt.plot(fpr_test, tpr_test)
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    path_test = os.path.join("/home/m_hamshi/results/", f"{file_name}_ROC_test.png")
    plt.savefig(path_test)
    plt.close()
    # plt.show() 


def create_cm_plot_val(confusion_matrix):
    file_name = current_file_name
    
    ax= plt.subplot()
    
    group_counts = ['{0:0.0f}'.format(value) for value in confusion_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues');

    # labels, title and ticks
    ax.set_title('Confusion Matrix') 

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    ax.xaxis.set_ticklabels(['Benign', 'Malignant']); 
    ax.yaxis.set_ticklabels(['Benign', 'Malignant']);

    path = os.path.join("/home/m_hamshi/results/", f"{file_name}_confusionMatrix_val.png")
    plt.savefig(path)
    plt.close()
    # plt.show()


def create_cm_plot_test(confusion_matrix):
    file_name = current_file_name
    
    ax= plt.subplot()
    
    group_counts = ['{0:0.0f}'.format(value) for value in confusion_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues');

    # labels, title and ticks
    ax.set_title('Confusion Matrix') 

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    ax.xaxis.set_ticklabels(['Benign', 'Malignant']); 
    ax.yaxis.set_ticklabels(['Benign', 'Malignant']);

    path = os.path.join("/home/m_hamshi/results/", f"{file_name}_confusionMatrix_test.png")
    plt.savefig(path)
    plt.close()
    # plt.show()

def plot_confusionMatrix_val():
    cm = metrics.confusion_matrix(val_trues, np.rint(val_preds))
    create_cm_plot_val(cm)

def plot_confusionMatrix_test():
    cm = metrics.confusion_matrix(test_trues, np.rint(test_preds))
    create_cm_plot_test(cm)


def save_classification_report():
    file_name = current_file_name 
    
    clf_report_val = pd.DataFrame(metrics.classification_report(y_true=val_trues, y_pred=np.rint(val_preds), output_dict=True)).transpose()
    clf_report_test = pd.DataFrame(metrics.classification_report(y_true=test_trues, y_pred=np.rint(test_preds), output_dict=True)).transpose()
    
    path_val = os.path.join("/home/m_hamshi/results/", f"{file_name}_clfReport_val.csv")
    path_test = os.path.join("/home/m_hamshi/results/", f"{file_name}_clfReport_test.csv")

    clf_report_val.to_csv(path_val, index= True)
    clf_report_test.to_csv(path_test, index= True)


# plot_learningCurve(history, 100)
plot_confusionMatrix_val()
plot_confusionMatrix_test()
# plot_roccurve()
save_classification_report()
