"""
    Image Processing helper functions
    © All rights reserved.
    author: spdkh
    date: 2023, JacobsSensorLab
"""
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler as Scaler

from src.utils import consts, geo_helper


def choose_random_images(
    n_imgs: int,
    imgs_paths: List[str],
    output_dir: str = 'sample_images',
    obj=None,  # module
    seed: int = None
    ):
    """
    This function randomly selects a specified number of images
    from a given list of image paths and plots them in a grid format.
    Parameters:
        - n_imgs (int): Number of images to be randomly selected.
        - imgs_paths (list): List of image paths: n * x * y (* ch)
        - output_dir (str): Directory to save the plotted images.
                            Default is 'sample_images'.
        - obj (module): Module used to read the images.
                        Default is None.
        - seed (int): Seed for the random selection.
                    Default is None.
    Returns:
        - imgs (list): List of randomly selected images.
        - titles (list): List of titles for each image.
    Processing Logic:
        - Randomly selects images from given paths.
        - Plots images in a grid format.
        - Uses given module to read images.
        - Uses given seed for random selection.

    """

    np.random.seed(seed)
    rand_samples = np.random.choice(len(imgs_paths),
                                    n_imgs, replace=False)
    my_imread = imread if obj is None else obj.imread

    imgs = [my_imread(imgs_paths[idx]) for idx in rand_samples]

    # The title for each image, extracting relevant parts
    titles = ['\n'.join(imgs_paths[idx].split('/')[-1][:-4].split('_')[2:4])\
        for idx in rand_samples]

    plot_multy(imgs, output_dir, n_imgs, 1,
               titles=titles)
    return imgs, titles


def plot_multy(
    imgs: List[np.ndarray],
    output_dir: str,
    cols: int,
    rows: int = 1,
    titles: List[str] = None
    ) -> None:
    """
    Plots multiple images in a grid layout.
    Parameters:
        - imgs (list): List of images to be plotted: n * x * y (* ch)
        - output_dir (str): Directory to save the plotted images.
        - cols (int): Number of columns in the grid layout.
        - rows (int): Number of rows in the grid layout.
                    Default is 1.
        - titles (list): List of titles for each image.
                        Default is None.
    Returns:
        - None: The function does not return any value.
    Processing Logic:
        - Creates a grid layout with specified number of rows and columns.
        - Displays each image in the grid layout.
        - Adds a title to each image if titles are provided.
        - Saves the plotted images in the specified output directory.
        - Closes all figures to avoid memory leak.
        """
    output_dir = str(output_dir)
    _, ax = plt.subplots(nrows=rows, ncols=cols,
                        figsize=(cols * 4,rows * 4),
                        subplot_kw=dict(xticks=[], yticks=[]))

    # Adjust layout to add space on the left for the title
    plt.subplots_adjust(left=0.2)

    if rows == 1:
        ax = np.expand_dims(ax, 0)

    print(len(imgs[0].shape))
    cmap = 'gray' if len(imgs[0].shape) == 2 else None
    # iterate over each row and column in the grid and display an image
    # The images to be displayed are accessed from the variable x
    i = 0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].imshow(imgs[i], cmap=cmap)
            if titles is None:
                ax[row, col].set_title(str(row)+str(col),
                                       fontsize=14)
            else:
                ax[row, col].set_title(titles[i], fontsize=14)
            i += 1

    # Add title to the left of y-axis and rotate it by 90 degrees
    plt.figtext(0.1, 0.5, output_dir.split('/')[-1],
                va='center', ha='center', rotation=90, fontsize=16)

    # show the figure
    plt.savefig(output_dir)  # Save sample results
    plt.show()
    plt.close("all")  # Close figures to avoid memory leak


def find_random_sample(
    n_images: int, labels: np.ndarray, data: object
    ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Finds a random sample from a dataset and returns its index, label, and coordinates.
    Parameters:
        - n_images (int): Number of images in the dataset.
        - labels (array): Array of labels for each image.
        - data (object): Object containing information about the dataset.
    Returns:
        - idx (int): Index of the randomly selected image.
        - label (array): Label of the randomly selected image.
        - coords (array): Coordinates of the bounding box for the label.
    Processing Logic:
        - Choose random index from dataset.
        - Normalize label values.
        - Calculate bounding box coordinates.
        - Return index, label, and coordinates.

    """
    idx = np.random.choice(n_images)
    cols = data.data_info['ytrain'].columns
    label = data.scaler.inverse_transform(
        labels.loc[idx, cols]
    )
    coords = geo_helper.calc_bbox_api(
        label[:2],
        label[2],
        data.input_dim
    )
    return idx, label, coords


def imread(img_path, shape=consts.IMG_SIZE[:2]):
    """
    Reads and normalizes images with specified dimension.
    Parameters:
        - img_path (string): Path to the image file.
        - shape (2D tuple of int): Desired shape of the image. Defaults to IMG_SIZE[:2].
    Returns:
        - img (type): Normalized image as a numpy array.
    Processing Logic:
        - Convert image to RGB if not already.
        - Resize image to specified shape.
        - Normalize image using min-max normalization.
    """
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(shape)
        img = Scaler().fit_transform(np.asarray(img))
    return img


def save_sample_data(data,
                    sample_label,
                    sample_img_name,
                    n_sample_imgs = 2):

    """
    Save sample data and corresponding labels as images for visualization purposes."
    Parameters:
        - data (list): List of sample data to be saved.
        - sample_label (list): List of labels corresponding to the sample data.
        - sample_img_name (str): Name of the sample image.
        - n_sample_imgs (int): Number of sample images to be saved. Default is 2.
    Returns:
        - None.
    Processing Logic:
        - Create a unique name for each sample image.
        - Convert sample data to numpy array.
        - Plot and save the sample image.
        - Repeat for specified number of sample images.
    """
    for i in range(n_sample_imgs):
        label = sample_label[i]
        sample_name = sample_img_name

        sample_name += '_' + str(i) + '_' + str(label) + '.png'
        sample_dir = consts.SAMPLE_DIR / sample_name

        sample_img = np.asarray(data[i])
        plt.figure()
        plt.imshow(sample_img)
        plt.title(sample_img_name + ' ' + str(i) + ' ' + str(label))
        plt.savefig(sample_dir)  # Save sample results
