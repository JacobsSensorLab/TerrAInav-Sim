"""
    Main Script to Run the program,
    load and configure the Google Map dataset
    © All rights reserved.
    author: spdkh
    date: July 2024, JacobsSensorLab
"""
import sys
import pandas as pd
import numpy as np
from src.utils import consts
from src.data.terrainav import terrAInav
from src.utils import geo_helper, io_helper

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

    coords_list = np.asanyarray(pd.read_csv(args.coords,
                                            dtype=float, sep=' '))

    for i, coords in enumerate(coords_list):
        io_helper.pretty(coords, header = 'Coordinate ' + str(i + 1))
        bbox_m = geo_helper.get_map_dim_m(
                args.fov, coords[-1],
                args.aspect_ratio[0]/args.aspect_ratio[1]
                )
        bbox = geo_helper.calc_bbox_m(coords[:2],
                                        bbox_m)
        args.coords = tuple(np.array(bbox).flatten()) + (coords[-1],)
        aerial_data = terrAInav(
        args=args,
        map_type=args.map_type,
        data_dir=args.data_dir,
        overlap=args.overlap
        )
        aerial_data.check_data()


if __name__ == '__main__':
    try:
        main()
    except Exception as Argument:
        print(Argument)
