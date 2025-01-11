"""
    Main Script to Run the program,
    load and configure the Google Map dataset
    © All rights reserved.
    author: spdkh
    date: July 2024, JacobsSensorLab
"""
import sys
import numpy as np
import pandas as pd
from src.utils import consts
from src.data.skyai import SkyAI
from src.utils import geo_helper

print("Command-line arguments:", sys.argv)

if "--help" in sys.argv:
    sys.exit(0)

def main():
    """Function:
        Generates a SkyAI object with specified parameters.
    Parameters:
        - args (list): List of arguments for the SkyAI object.
        - map_type (str): Type of map to be generated (default: 'satellite').
        - data_dir (str): Directory to store the generated map data (default: current directory).
        - overlap (int): Amount of overlap between adjacent map tiles (default: 0).
    Returns:
        - aerial_data (SkyAI): SkyAI object with specified parameters.
    Processing Logic:
        - Generate SkyAI object.
        - Set map type to 'satellite' if not specified.
        - Set data directory to current directory if not specified.
        - Set overlap to 0 if not specified.
    """
    args = consts.ARGS

    # if coords argument was given as a file instead of the actual coordinates:
    try:
        args.coords = np.asanyarray(pd.read_csv(args.coords, dtype=float, sep=' '))[0]
    except:
        pass

    bbox_m = geo_helper.get_map_dim_m(
        args.fov, args.coords[-1],
        args.aspect_ratio[0]/args.aspect_ratio[1]
        )

    bbox = geo_helper.calc_bbox_m(args.coords[:2],
                                  bbox_m)

    args.coords = tuple(np.array(bbox).flatten()) + (args.coords[-1],)
    aerial_data = SkyAI(
        args=args,
        map_type=args.map_type,
        data_dir=args.data_dir,
        overlap=args.overlap
        )
    aerial_data.data_info['x'] = '.'
    aerial_data.check_data()


if __name__ == '__main__':
    main()
