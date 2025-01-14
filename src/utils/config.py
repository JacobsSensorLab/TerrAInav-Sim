"""
    parsing and configuration
    © All rights reserved.
    author: spdkh
    date: May 10, 2022
"""
import os
import argparse
import json

from src.utils.io_helper import pretty, str_to_floats


def parse_args():
    """
        Define terminal input arguments
    Returns
    -------
    arguments
    """

    parser = argparse.ArgumentParser(
        prog='SkyAI Sim Project',
        description='Aerial Imaging Simulation from Satellite Data'
        )
    # Define the arguments in a dictionary
    arguments = {
        'coords': {
            'type': str_to_floats,
            'default': "35.16_-89.90_35.115_-89.823_120",
            'help':(
            'Input can be either a file or a formatted string.\n'
            '1. If a string is entered, it should refer to a file with the following format (lat lon agl(feet)):\n'
            '   34.052235 -118.243683 100\n'
            '   40.712776 -74.005974 50\n'
            '   51.507351 -0.127758 200\n'
            '   If a single download file is run, it will only take into account the first row.\n'
            '2. The input can also be a string in the following format: TopLeftLat_TopLeftLon_BottoRightLat_BottomRightLon_AGL(f)\n'
            '   e.g., "35.22_-90.07_35.06_-89.73_400"\n'
            '   For single download, it will take the top-left coordinate as the central coordinates and discard the bottom-right.\n'
            '3. Alternatively, input can be only the central coordinates and agl: "lat_lon_agl"\n')
        },
        'fov': {
            'type': float,
            'default': 78.8,
            'help': 'Diagonal field of view of the camera in degrees.'
        },
        'aspect_ratio': {
            'type': float, 'nargs': '+',
            'default': [4, 3],
            'help': 'Aspect ratio dimensions.'
        },
        'map_type':{
            'type':str,
            'default': 'satellite',
            'help': 'Map Type from static map API',
            'choices': ['satellite', 'roadmap', 'terrain']
        },
        'dataset':{
            'type': str,
            'default': 'SkyAI',
            'help': 'Specify the name of the datset.',
            'choices': ['SkyAI', 'VBN']
        },
        'data_dir': {
            'type': str,
            'default': 'dataset/Memphis/',
            'help': 'Directory name to save the generated images.'
        },
        'img_size': {
            'type': int, 'nargs': '+',
            'default': [400, 400, 3],
            'help': 'The desired image size to resize to after loading the original image.'
        },
        'overlap': {
            'type': float,
            'default': 0,
            'help': 'Overlap of the camera field of view as a probability value between 0 to 1.'
        },
        'batch_size': {
            'type': int,
            'default': 8,
            'help': 'The size of batch (only for machine learning tasks.)'
        },
        'seed': {
            'type': int,
            'default': 2024,
            'help': 'Random seed value.'
        }
    }
    # Load defaults from JSON and overwrite initial defaults if present
    json_defaults = update_args_with_json('src/utils/config.json')
    for arg, value in json_defaults.items():
        if arg in arguments:
            arguments[arg]['default'] = value

    # Add arguments to the parser with their default values
    for arg, options in arguments.items():
        parser.add_argument(f'--{arg}', **options)

    return parser


def update_args_with_json(json_file):
    """
    Updates default arguments with values from a JSON file.
    Parameters:
        - args_ (argparse.Namespace):
            Namespace object containing arguments.
        - json_file (str): Path to JSON file.
    Returns:
        - None: The function does not return anything.
    Processing Logic:
        - Load JSON file.
        - Iterate through key-value pairs.
        - Set attribute for each key-value pair.
    """
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            pretty("Error loading JSON file:" + e, header='Warning!')
            return {}
    else:
        pretty("Default file" ,json_file, "not found.",
               header='Warning!')
        return {}
