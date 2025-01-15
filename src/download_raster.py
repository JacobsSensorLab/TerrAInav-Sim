"""
    Main Script to Run the program,
    load and configure the Google Map dataset
    © All rights reserved.
    author: spdkh
    date: June 2024, JacobsSensorLab
"""
import sys
from src.utils import consts
from src.data.terrainav import terrAInav

if "--help" in sys.argv:
    sys.exit(0)

def main():
    """Function:
        Generates a terrAInav object with specified parameters.
    Parameters:
        - args (list): List of arguments for the terrAInav object.
        - map_type (str): Type of map to be generated (default: 'satellite').
        - data_dir (str): Directory to store the generated map data (default: current directory).
        - overlap (int): Amount of overlap between adjacent map tiles (default: 0).
    Returns:
        - aerial_data (terrAInav): terrAInav object with specified parameters.
    Processing Logic:
        - Generate terrAInav object.
        - Set map type to 'satellite' if not specified.
        - Set data directory to current directory if not specified.
        - Set overlap to 0 if not specified.
    """
    args = consts.ARGS

    aerial_data = terrAInav(
        args=args,
        map_type=args.map_type,
        data_dir=args.data_dir,
        overlap=args.overlap
        )
    aerial_data.config(download_raster=True)

    # To remove data with no available features
    # The threshold will be selected experimentally
    # checkout src/notebooks/datacleaner example
    aerial_data.cleanup_data(entropy_thr=2.1)

    # Prepar the dataset for a keras DNN task
    # augmented_data = aerial_data.config_dnn()


if __name__ == '__main__':
    main()
