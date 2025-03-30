"""
    Vision Based Navigation Data
    © All rights reserved.
    author: spdkh
    date: June 2023, JacobsSensorLab
"""
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler
from keras import layers
from keras.models import Sequential
import tensorflow as tf

from src.utils import consts, io_helper, geo_helper, img_helper
from src.utils.io_helper import pretty
from src.data.imagedata import ImageData

class VBN(ImageData):
    """
    Class VBN (Vision Based Navigation):
        This class inherits from the ImageData abstract class
            and represents a specific dataset manager for the VBN dataset.

        Attributes:
            - scaler: An instance of the MinMaxScaler used for scaling the output labels.
            - data_types: A dictionary specifying the types of data
                ('x' for images, 'y' for metadata).
            - data_dir: The directory where the dataset is located.
            - input_dim: The dimensions of the input images.
            - output_dim: The dimension of the output labels.
            - input_dir: A list of paths to the input image files.
            - output_dir: A list of paths to the output metadata files.
            - img_augmentation: A Sequential model for image augmentation.

        Methods:
            - __init__: Initializes the VBN dataset manager with the specified arguments.
            - config_output:
                Configures the output data by reading metadata files and extracting labels.
            - config_dnn: Configures the deep neural network (DNN) by preparing the dataset,
                splitting it, and saving sample data.
            - my_train_test_split: Splits the dataset into train, test, and validation sets
                and normalizes the outputs.
            - keras_dataset: Creates a TensorFlow dataset for training or testing.
            - preprocess_image: Preprocesses an image by converting it to a float32 tensor
                and resizing it.
            - preprocess_label: Preprocesses the labels (not implemented in this class).

        Note:
            This class is specific to the VBN dataset and provides functionality
            for data loading, preprocessing, and dataset preparation for deep learning tasks.
    """
    def __init__(self, **kwargs):
        """
            args
        """
        super().__init__(**kwargs)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.scaler = Scaler()
        self.data_types = {'x': 'JPG2', 'y': 'MetaData2'}
        data_dir = kwargs['data_dir'] if 'data_dir' in kwargs\
            else "/home/sdjkhosh/Datasets/VisnavPNGFiles/jpg Simulated Files/Raster"
        self.data_dir = Path(data_dir)

        self.input_dim = self.args.img_size
        self.output_dim = 3

        self.input_dir = io_helper.find_files(self.data_dir /
                                                 self.data_types['x'],
                                                 'jpg')

        self.output_dir = io_helper.find_files(self.data_dir, 'txt')
        self.img_augmentation = Sequential(
        [
            layers.RandomRotation(factor=0.25, fill_mode='reflect'),
            # layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            # layers.RandomContrast(factor=0.05),
            # layers.RandomBrightness(factor=0.05),
            layers.RandomZoom(height_factor=(-0.3, 0.3), fill_mode='reflect')
        ],
        name="img_augmentation",
        )

    def config_output(self):
        self.output_dir.sort()
        meta_df = pd.read_csv(self.output_dir[0], sep=':', index_col=0,
                              names=[0])

        pretty('Number of texts in the path:', len(self.output_dir),
               '\nFirst metadata sample:\n', meta_df, log=self)

        meta_dfs = []
        for i, meta_data in enumerate(self.output_dir[:-1]):
            df_i = pd.read_csv(meta_data, sep=':', index_col=0, names=[i + 1])
            meta_dfs.append(df_i.iloc[:, 0])

        meta_df = pd.concat(meta_dfs, axis=1)

        pretty('All metadata:\n', meta_df, log=self)

        self.labels = meta_df.loc['Platform_position_LatLongAlt', :]
        self.labels = \
            self.labels.str.split(" ",
                                        expand=True).iloc[:, 1:-1].astype('float64')
        self.labels.columns = ['Lat', 'Lon', 'Alt']

    def config_dnn(self, n_sample_imgs=2):
        """Function:
        Configure DNN preparation, datasplit, and sampling
        Parameters:
            - self (object): Instance of the class.
            - n_sample_imgs (int): Number of sample images to save in each mode.
        Returns:
            - img_augmentation (Sequential): A sequential model for image augmentation.
        Processing Logic:
            - Saves sample data for each mode.
            - Applies random rotation, flip, and zoom to the sample data.
            - Saves the augmented sample data.
            - Uses tqdm to track progress.
        """
        self.my_train_test_split()

        pretty('Saving sample data...')
        with tqdm(position=0, leave=True, total=n_sample_imgs * len(self.modes)) as pbar:
            io_helper.check_folder(consts.SAMPLE_DIR)
            for mode in self.modes:
                self.dataset[mode] = self.keras_dataset(mode)

                sample_data = list(self.dataset[mode].take(1).as_numpy_iterator())[0]
                sample_img_name = 'sample_'+ mode
                img_helper.save_sample_data(sample_data[0],
                                       sample_data[1],
                                         sample_img_name)

                img_augmentation = Sequential(
                    [
                        layers.RandomRotation(factor=0.25, fill_mode='reflect'),
                        # layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                        layers.RandomFlip(),
                        # layers.RandomContrast(factor=0.05),
                        # layers.RandomBrightness(factor=0.05),
                        layers.RandomZoom(height_factor=(-0.3, 0.3), fill_mode='reflect')
                    ],
                    name="img_augmentation",
                    )
                aug_img = img_augmentation(sample_data[0])
                sample_aug_name = 'aug_sample_'+ mode
                img_helper.save_sample_data(aug_img,
                                       sample_data[1],
                                         sample_aug_name)
                pbar.update()
        return img_augmentation

    def my_train_test_split(self):
        """
        Split the dataset into train, test, and validation sets and normalize the outputs.
        Parameters:
            - self: The current object.
        Returns:
            - None.
        Processing Logic:
            - Convert labels to UTM coordinates.
            - Normalize the outputs.
            - Split the dataset into train, test, and validation sets.
            - Print the sample image size.

        todo: check scaler should apply scaling per column
        """
        updated_labels = np.asarray([geo_helper.geo2utm(lat, lon, self.args.utm) \
            for lat, lon in self.labels.loc[:, ['Lat', 'Lon']].values])

        pretty('Sample Network Outputs\n:',
               updated_labels[:10], log=self)
        y_normalized = pd.DataFrame(
            self.scaler.fit_transform(updated_labels),
            columns=['Lat', 'Lon']
        )
        y_normalized['Alt'] = self.labels['Alt']

        pretty('Normalized outputs (y_normalized):\n', y_normalized, log=self)

        # Split to 80 train, 10 test, 10 validation
        self.data_info['x'+ self.modes[0]], x_test, \
        self.data_info['y'+ self.modes[0]], y_test \
            = train_test_split(self.input_dir, y_normalized,
                               test_size=0.2,
                               random_state=self.args.seed,
                               shuffle=False)
        self.data_info['x'+ self.modes[1]], \
                    self.data_info['x'+ self.modes[2]], \
                        self.data_info['y'+ self.modes[1]], \
                            self.data_info['y'+ self.modes[2]] \
                                = train_test_split(x_test, y_test,
                                                    test_size=0.5,
                                                    random_state=self.args.seed,
                                                    shuffle=False)

        text = ''
        for mode in self.modes:
            for status in ['x', 'y']:
                text += '\n'+ status + '_' + mode
                text += ' size: ' + str(self.data_info[status + mode])
        pretty('Sample image size:', self.input_dim, log=self)

    def keras_dataset(self, mode):
        """Parameters:
            - mode (str): Determines which dataset to use, either 'train' or 'test'.
        Returns:
            - dataset (tf.data.Dataset): A preprocessed dataset ready for training or testing.
        Processing Logic:
            - Creates a dataset from the provided data.
            - Applies preprocessing to the dataset.
            - Batches the dataset according to the provided batch size.
            - Prefetches the dataset for faster processing.

        References:
            - https://www.tensorflow.org/tutorials/load_data/images
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data_info['x' + mode],
            self.data_info['y' + mode])
        )
        dataset = dataset.map(self.preprocess,
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.args.batch_size,
                                drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def preprocess_image(self, filename):
        """
        Preprocesses an image by converting it to a float32 tensor
        and resizing it to the input dimensions.
        Parameters:
            - filename (str): The name of the image file to be preprocessed.
        Returns:
            - image (tensor): A float32 tensor of the preprocessed image.
        Processing Logic:
            - Read image file as string.
            - Decode image as JPEG with 3 channels.
            - Convert image to float32.
            - Resize image to input dimensions.
        """
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.input_dim[0:2])
        return image

    def preprocess_label(self, label):
        return label

    def imread(self, img_path):
        '''
            Read Tensorflow image
        '''
        image_string = tf.io.read_file(str(img_path))
        image = tf.image.decode_jpeg(image_string, channels=self.input_dim[-1])
        image = tf.image.convert_image_dtype(image, tf.uint8)

        return image
    