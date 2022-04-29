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
from tensorflow.keras.applications import VGG16

# The path to the directory of the program dataset
base_dir = "/home/m_hamshi/Dataset/Break_data/"

current_file_name = Path(__file__).stem

# Directories for training, validation and test splits
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# conv_base = models.load_model("/home/m_hamshi/models/InceptionResNetV2_conv_base_128.h5")

conv_base = VGG16(weights=None, include_top=True,
                  input_shape=(128, 128, 3))


conv = conv_base.layers[-2]
fc1 = Dense(64, activation='relu')
dropout = Dropout(0.1)
fc2 = Dense(1, activation='sigmoid')

# Reconnect the layers
x = fc1(conv.output)
x = dropout(x)
predictors = fc2(x)

# Create a new model
base_model = Model(conv_base.input, predictors)


def save_log_csv(file_name):
    # path = os.path.join("./results/", f"{file_name}.csv")
    path = os.path.join("/home/m_hamshi/results/", f"{file_name}.csv")
    history_logger = tf.keras.callbacks.CSVLogger(
        path, separator=",", append=True)
    return history_logger


history_logger = save_log_csv(current_file_name)

# model = Sequential()
# model.add(conv_base)
# model.add(Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(1, activation='sigmoid'))

# conv_base.trainable = False


def preprocess_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=(0.8, 1.2)
    )

    # All images will be rescaled by 1./255
    # train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(128, 128),
        color_mode="rgb",
        batch_size=100,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        color_mode="rgb",
        batch_size=50,
        class_mode='binary',
        shuffle=False)

    return train_generator, validation_generator


train_generator, validation_generator = preprocess_data()


base_model.compile(loss='binary_crossentropy',
                   optimizer=optimizers.Adam(learning_rate=1e-5),
                   # optimizer=optimizers.RMSprop(lr=2e-3),
                   metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

history = base_model.fit_generator(
    train_generator,
    steps_per_epoch=300,
    epochs=100,
    callbacks=[history_logger],
    validation_data=validation_generator,
    validation_steps=12
)


predictions = base_model.predict_generator(validation_generator, steps=12)
val_preds = predictions
val_trues = validation_generator.classes


# conv_base.trainable = True

# set_trainable = False

# for layer in conv_base.layers:
#     if layer.name == 'block4_conv3':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False


# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.Adam(learning_rate=1e-5),
#               # optimizer=optimizers.RMSprop(lr=2e-3),
#               metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=12,
#     epochs=200,
#     callbacks=[history_logger],
#     validation_data=validation_generator,
#     validation_steps=12
# )

base_model.save_weights("/home/m_hamshi/models/weights_VGG16_Pre.h5")


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

    fpr , tpr , thresholds = metrics.roc_curve(val_trues, val_preds)

    dataframe1 = pd.DataFrame({'val_trues': np.ravel(val_trues), 'val_preds': np.ravel(val_preds)})
    dataframe2 = pd.DataFrame({'fpr': np.ravel(fpr), 'tpr': np.ravel(tpr)})
    csv_path1 = os.path.join("/home/m_hamshi/results/", f"{file_name}_preds.csv")
    csv_path2 = os.path.join("/home/m_hamshi/results/", f"{file_name}_fpr.csv")
    dataframe1.to_csv(csv_path1) 
    dataframe2.to_csv(csv_path2) 

    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    path = os.path.join("/home/m_hamshi/results/", f"{file_name}_ROC.png")
    plt.savefig(path)
    plt.close() 
    # plt.show() 


def create_cm_plot(confusion_matrix):
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

    path = os.path.join("/home/m_hamshi/results/", f"{file_name}_confusionMatrix.png")
    plt.savefig(path)
    plt.close()
    # plt.show()


def plot_confusionMatrix():
    cm = metrics.confusion_matrix(val_trues, np.rint(val_preds))
    create_cm_plot(cm)


def save_classification_report():
    file_name = current_file_name 
    
    clf_report = pd.DataFrame(metrics.classification_report(y_true=val_trues, y_pred=np.rint(val_preds), output_dict=True)).transpose()
    
    path = os.path.join("/home/m_hamshi/results/", f"{file_name}_clfReport.csv")
    clf_report.to_csv(path, index= True)


plot_learningCurve(history, 200)
plot_confusionMatrix()
plot_roccurve()
save_classification_report()