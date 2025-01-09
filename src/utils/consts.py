"""
    © All rights reserved.
    author: spdkh
    date: sprint 2024, JacobsSensorLab
"""
import math
from pathlib import Path
from turtle import update
from src.utils.config import parse_args
from src.utils.io_helper import str_to_floats


try:
    args = parse_args().parse_args()
except:
    # To be able to run in google colab
    args = parse_args().parse_args(args=[])
args.coords = str_to_floats(args.coords)
ARGS = args

# the path corresponding to current file directory
THIS_DIR = Path(__file__).__str__()
# the path corresponding to source path
SRC_DIR = Path(THIS_DIR).parents[1]
PROJ_DIR = Path(THIS_DIR).parents[2]
CONFIG_DIR = SRC_DIR / 'config'

SAMPLE_DIR = PROJ_DIR / 'sampled_ml_imgs'

IMG_SIZE = (400, 400, 3)

# center of mercator projection tile in pixels
tile_size = 256
tile_center_p = {'x': tile_size/2, 'y': tile_size/2}

# number of pixels per degree of longitude at zoom 0
pixel_per_degree = 256 / 360

# The mercator projection stretches the earth's surface into a flat map.
# This projection results in a vertical (y-axis) scaling factor
pixel_per_radian = 256 / (2 * math.pi)
