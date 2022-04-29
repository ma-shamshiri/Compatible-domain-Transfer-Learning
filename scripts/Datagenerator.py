import os
from keras.preprocessing.image import ImageDataGenerator


def create_directory(path):
    """
    Creates a directory in the given path if does not exist.

    -> Args:
        path (str): A string representing the directory path
    """

    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)


def count_files(dir_path):
    count = 0

    for directories, subdirectories, files in os.walk(dir_path):
        for filename in files:
            if filename.split('.')[-1] == 'png':
                count += 1

    return count


def augment_data(src_path, dst_path, num_files):
    datagen = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=(0.8, 1.2)
    )

    train_generator = datagen.flow_from_directory(
        src_path,
        target_size=(128, 128),
        color_mode="rgb",
        batch_size=num_files,
        save_to_dir=dst_path,
        class_mode="binary",
        save_prefix="aug",
        save_format="png")

    for i in range(50):
        train_generator.next()


src_path = "/home/m_hamshi/Dataset/New_m"
dst_path = "/home/m_hamshi/Dataset/Break_data/train/malignant"

create_directory(dst_path)
num_files = count_files(src_path)
augment_data(src_path, dst_path, num_files)
