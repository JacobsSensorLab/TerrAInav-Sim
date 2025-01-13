"""
    Geolocation helper functions
    © All rights reserved.
    author: spdkh
    date: Aug 2023, JacobsSensorLab
"""
from typing import Tuple, List
import math
import time
from io import BytesIO

from geopy.distance import geodesic
from geopy.point import Point
from matplotlib.streamplot import OutOfBounds
import pyproj
import numpy as np
import tensorflow as tf
import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw


from src.utils import consts
from src.utils.io_helper import pretty


def get_map_image(response, data_dir: str, retry: int=10
    ):
    """Get a static map image from Google Maps API given
    latitude and longitude coordinates, map type, zoom level, size, and API key if available.
    Parameters:
        - data_dir (str): Direction where to save the loaded map data including datatype
        - retry (int): Number of times to retry dowloading if errors happened
    Processing Logic:
        - Saves the collected data to the given directory if available
        - Otherwise, specify the error and the URL for the error
    """
    final_url = response.url

    for i in range(retry):
        if response.status_code == 200:
            with open(data_dir, "wb") as f:
                f.write(response.content)
            return
    if response.status_code != 200:
        raise ValueError(
            "Trial", i,
            "\nFailed to retrieve the image. Status code:", response.status_code,
            "URL:", final_url[:-1]
            )


def init_static_map(coords: Tuple[float, float],
    map_type: str, zoom: int=15, size: Tuple[int, int]=(640, 640),
    api_key: str=None):
    """Get a static map image from Google Maps API given
    latitude and longitude coordinates, map type, zoom level, size, and API key if available.
    Parameters:
        - coords (tuple): Latitude and Longitude coordinate.
        - map_type (str): Type of map to retrieve.
        - zoom (int): Zoom level of the map, default is 15.
        - size (tuple): Size of the map image in pixels, default is (640, 640).
        - api_key (str): API key for Google Maps API, if available.
    Processing Logic:
        - Imports API key from hidden_file.py if available.
        - Constructs base URL and parameters for API call.
        - Removes labels from the map image.
        - Constructs final URL for API call.
        - Makes API call using requests library.
    """
    try:
        if api_key is None:
            from src.hidden_file import api_key
    except ModuleNotFoundError:
        pretty('hidden_file.py is not available.',
               header='Warning!')
        api_key = '<API-Key>'

    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{coords[0]},{coords[1]}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[1]}",
        "maptype": map_type,
        "style": "feature:all|element:labels|visibility:off",  # Remove labels
        "key": api_key,
    }

    if api_key == '<API-Key>':
        return base_url + "?" \
        + "&".join([f"{key}={value}" for key, value in params.items()])
    return requests.get(base_url, params=params)


def collect_tiles(tl, br, zoom, map_type, resolution):
    """
    Stitch a map image by collecting and assembling map tiles for a given geographical bounding box.
    Parameters:
        - tl (tuple): The coordinates (latitude, longitude) of the top-left corner.
        - br (tuple): The coordinates (latitude, longitude) of the bottom-right corner.
        - zoom (int): The zoom level for the map tiles.
        - map_type (str): Type of map ('satellite', 'roadmap', or 'terrain').
        - resolution (int): Additional resolution level to increase detail.
    Returns:
        - Image: A PIL Image object of the stitched map covering the specified region.
    Example:
        - collect_tiles((37.7749, -122.4194), (37.7649, -122.4094), 15, 'roadmap', 0) -> <PIL.Image.Image image mode=RGB size=XxY at 0x...>
    Notebook: https://colab.research.google.com/drive/1p9kFCQbDPRsJJM6RXsrsNySCSGSrTnFc?usp=sharing
    """
    def generate_tile_info(loc, zoom):
        """
        Calculate tile coordinates and pixel position for mapping services.
        Parameters:
            - loc (tuple): A tuple containing latitude and longitude coordinates.
            - zoom (int): The zoom level for which the tile information is calculated.
        Returns:
            - tuple: A tuple containing the tile x-coordinate, tile y-coordinate, pixel x offset, and pixel y offset.
        Example:
            - generate_tile_info((37.7749, -122.4194), 10) -> (163, 395, 103, 120)
        """
        # Calculate tile coordinates
        n = 2 ** zoom
        lat, lon = loc
        tile_x = (lon + 180.0) / 360.0 * n
        tile_y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n

        pixel_x = int(tile_x * 256 - int(tile_x) * 256)
        pixel_y = int(tile_y * 256 - int(tile_y) * 256)

        # Generate URL s&x for sat, m&x for roadmap
        return int(tile_x), int(tile_y), pixel_x, pixel_y
    def fetch_and_paste(tile_id_x, tile_id_y, map_type, max_retries=10):
        """
        Fetches a map tile from a given map type and pastes it into a larger stitched image.
        Parameters:
            - tile_id_x (int): X-coordinate of the tile to be fetched.
            - tile_id_y (int): Y-coordinate of the tile to be fetched.
            - map_type (str): Type of map to fetch the tile from ('satellite', 'roadmap', 'terrain').
            - max_retries (int, optional): The maximum number of retries for fetching the tile. Defaults to 10.
        Returns:
            - None
        Example:
            - fetch_and_paste(10, 15, 'satellite', 5) -> None
        """
        map_types = {'satellite':'s', 'roadmap':'m', 'terrain':'t'}

        for attempt in range(1, max_retries + 1):
            try:
                url = f"https://mt.google.com/vt/lyrs={map_types[map_type]}&x={tile_id_x}&y={tile_id_y}&z={zoom}"
                response = requests.get(url)
                response.raise_for_status()

                # Load the image into Pillow
                image = Image.open(BytesIO(response.content))

                # Calculate position to paste the tile in the stitched image
                offset_x = (tile_id_x - tl_info[0]) * 256
                offset_y = (tile_id_y - tl_info[1]) * 256

                # Paste the tile onto the stitched image
                stitched_image.paste(image, (offset_x, offset_y))

                # Break the loop if successful
                break
            except Exception as e:
                if attempt < max_retries:
                    print(f"Attempt {attempt} failed for tile ({tile_id_x}, {tile_id_y}). Retrying...")
                    time.sleep(1)  # Optional: add a small delay between retries
                else:
                    print(f"Error fetching tile ({tile_id_x}, {tile_id_y}) after {max_retries} attempts: {e}")

    zoom = min(zoom + resolution, 22)

    # Calculate tile coordinates
    tl_info = generate_tile_info(tl, zoom)
    br_info = generate_tile_info(br, zoom)

    # Calculate dimensions of the stitched image
    n_tiles_x = br_info[0] - tl_info[0] + 1
    n_tiles_y = br_info[1] - tl_info[1] + 1

    img_w = abs(br_info[-2] + 256 * (n_tiles_x - 1)- tl_info[-2])
    img_h = abs(br_info[-1] + 256 * (n_tiles_y - 1)- tl_info[-1])

    stitched_image = Image.new("RGB",
                               (n_tiles_x * 256,
                                n_tiles_y * 256))
    with ThreadPoolExecutor() as executor:
        for tile_id_y in range(tl_info[1], br_info[1]+1):
            for tile_id_x in range(tl_info[0], br_info[0]+1):
                executor.submit(fetch_and_paste, tile_id_x, tile_id_y, map_type)

    cropped_image = stitched_image.crop((tl_info[-2],
                                         tl_info[-1],
                                         tl_info[-2] + img_w,
                                         tl_info[-1] + img_h))
    return cropped_image


def calc_bbox_api(
    center: Tuple[float, float], zoom: int,
    map_size: Tuple[int, int]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculates the bounding box coordinates for a given center point, zoom level, and map size.
    Parameters:
        - center (tuple):
            A tuple containing the latitude and longitude coordinates of the center point.
        - zoom (int):
            The zoom level of the map.
        - map_size (tuple):
            A tuple containing the width and height of the map in pixels.
    Returns:
        - tuple: A tuple containing the latitude and longitude coordinates of
                the top left and bottom right corners of the bounding box.
    Processing Logic:
        - Calculates the pixel size based on the zoom level.
        - Calculates the x and y coordinates of the center point.
        - Uses the x and y coordinates to calculate
            the top left and bottom right corners of the bounding box.
        - Converts the coordinates from pixel coordinates to latitude and longitude coordinates.
            The coordinates belong to top left and bottom right.

    Converted from JS to python using chatGPT
    Reference:
        https://stackoverflow.com/questions/44784839/calculate-bounding-box-of-static-google-maps-image
    """

    def clamp(value: float, min_value: float, max_value: float) -> float:
        """
        Ensures that a given value is within the specified range
        [min_value, max_value]
        """
        return max(min(value, max_value), min_value)

    def pt_to_lat_lon(pt: dict[str, float]) -> Tuple[float, float]:
        """
        Converts a point from pixel coordinates to latitude and longitude coordinates.
        """
        lon = (pt['x'] - consts.tile_center_p['x']) / consts.pixel_per_degree
        lat = math.degrees(math.asin(math.tanh((pt['y'] - consts.tile_center_p['y']) / -consts.pixel_per_radian)))
        return lat, lon

    # the width and height of the map in pixels, adjusted by the pixel_size
    # converted to mercator projection pixel values based on zoom
    pixel_size = pow(2, -zoom)
    half_pw_x = map_size[0] * pixel_size / 2
    half_pw_y = map_size[1] * pixel_size / 2

    a = clamp(
        math.sin(math.radians(center[0])),
        -(1 - 1E-15), 1 - 1E-15)

    # adjusted center point pixel coordinates
    cp = {
        'x': consts.tile_center_p['x'] + center[1] * consts.pixel_per_degree,
        'y': consts.tile_center_p['y'] + 0.5 * math.log((1 + a) / (1 - a)) * -consts.pixel_per_radian
    }

    top_left = pt_to_lat_lon({'x': cp['x'] - half_pw_x, 'y': cp['y'] - half_pw_y})
    bottom_right = pt_to_lat_lon({'x': cp['x'] + half_pw_x, 'y': cp['y'] + half_pw_y})

    return top_left, bottom_right


def get_zoom_from_bounds(
    top_left: Tuple[float, float],
    bottom_right: Tuple[float, float],
    zoom_bound=22
    ) -> Tuple[int, List[int]]:
    """
    Reverse the bounding box coordinates from lat/lon to zoom level and image size.
    If you know the top left and bottom right lat/lon,
    what would be the best zoom level that can fit the biggest possible image size (640) in it.
    Parameters:
        - top_left (tuple):
            Latitude and longitude of the top left corner of the bounding box.
        - bottom_right (tuple):
            Latitude and longitude of the bottom right corner of the bounding box.
        - zoom_bound (int): maximum accepted zoom level (based on map type)
                default is 22 (for satellite and roadmap)
    Returns:
        - zoom (int): Zoom level of the bounding box.
        - img_size (list): Image size of the bounding box in the format [width, height].
    Processing Logic:
        - Convert lat/lon coordinates to pixel points.
        - Calculate the pixel width and height of the bounding box.
        - Determine the zoom level based on the pixel width and height.
        - Calculate the final image size based on the zoom level.
    """
    def latlonToPt(lat, lon):
        a = min(
            max(
                math.sin(math.radians(lat)),
                -(1 - 1E-15)
                ),
            1 - 1E-15
            )
        cp = {
            'x': consts.tile_center_p['x'] + lon * consts.pixel_per_degree,
            'y': consts.tile_center_p['y'] + 0.5 * math.log((1 + a) / (1 - a)) * -consts.pixel_per_radian
        }
        return cp
    # Convert lat/lon coordinates to points
    cp_top_left = latlonToPt(top_left[0], top_left[1])
    cp_bottom_right = latlonToPt(bottom_right[0], bottom_right[1])

    # Calculate half pixel width and height
    half_pw_x = (cp_bottom_right['x'] - cp_top_left['x']) / 2
    half_pw_y = (cp_bottom_right['y'] - cp_top_left['y']) / 2
    # Initialize image size
    # This is the maximum pixel size available on Google Map
    # We get the high resolution first, then resize to the desired dimensions if needed
    img_size = consts.tile_size

    # The bigger width along x or y will be the img_size
    # The other width is adjusted based on aspect ratio
    # Determine zoom level
    zoom = int(-math.log2(max(half_pw_x, half_pw_y) / img_size) - 1)
    # Calculate final image width and height based on zoom level
    scaling_factor = 2 ** (zoom + 1)
    img_w = int(half_pw_x * scaling_factor)
    img_h = int(half_pw_y * scaling_factor)

    if zoom > zoom_bound:
        raise  OutOfBounds("Zoom Level", zoom, "is out of bounds.")

    return zoom, [img_w, img_h]


def overlapped(coords_a: tuple, coords_b: tuple, overlap: int = 25) -> bool:
    """Checking if two images with known label
        (top left and bottom right lat, long) are overlapped
        more than the desired overlap amount specified
    Parameters:
        - coords_a (tuple): Top left (lat, long) of the first image
        - coords_b (tuple): Buttom right(lat, lon) of the second image
        - overlap (int): [0-100] overlap percentage
    Returns:
        - bool: wether two images are overlapped by the percentage (True) or not (False)

    """
    return find_overlap(coords_a, coords_b) >= overlap


def find_overlap(coords_a: tuple, coords_b: tuple) -> int:
    """Calculates the percentage of overlap between two rectangles.
    Parameters:
        - coords_a (tuple): Coordinates of the first rectangle in the format
                            (top left lat, top left lon, bottom right lat, bottom right lon).
        - coords_b (tuple): Coordinates of the second rectangle in the format
                            (top left lat, top left lon, bottom right lat, bottom right lon).
    Returns:
        - overlap_percentage (int): Percentage of overlap between the two rectangles.
    Processing Logic:
        - Calculates the overlap in the x and y directions.
        - Calculates the area of each rectangle.
        - Calculates the percentage of overlap by dividing the overlap area
            by the total area of both rectangles.
        - Returns the overlap percentage as an int.
    """

    top_left_lat_a, top_left_lon_a, bottom_right_lat_a, bottom_right_lon_a = coords_a
    top_left_lat_b, top_left_lon_b, bottom_right_lat_b, bottom_right_lon_b = coords_b

    y_overlap = max(0, min(top_left_lat_a, top_left_lat_b)\
                    - max(bottom_right_lat_a, bottom_right_lat_b))
    x_overlap = max(0, min(bottom_right_lon_a, bottom_right_lon_b)\
                    - max(top_left_lon_a, top_left_lon_b))

    rect1_area = abs(bottom_right_lat_a - top_left_lat_a)\
                    * abs(bottom_right_lon_a - top_left_lon_a)
    rect2_area = abs(bottom_right_lat_b - top_left_lat_b)\
                    * abs(bottom_right_lon_b - top_left_lon_b)

    overlap_percentage =  x_overlap * y_overlap \
        / (rect1_area + rect2_area -  x_overlap * y_overlap)

    return int(overlap_percentage * 100)


def geo_calcs(data):
    """
        Gives information about the geolocation including
        minimum and maximum lat, long, alt, also
        area, width, height, etc in meters.
        The calculations are actual measurements based on the images already downloaded.
        the coordinates are based on the top left of the first image
        (slightly higher and lefter than top left of the map)
        and bottom right of the last image
        (slightly lower and righter than bottom right of the map)
    """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    coords_ul = (data_min['Lat'], data_min['Lon'])
    coords_ur = (data_max['Lat'], data_min['Lon'])
    coords_dl = (data_min['Lat'], data_max['Lon'])
    coords_dr = (data_max['Lat'], data_max['Lon'])

    land_width = geodesic(coords_ul, coords_ur).km
    land_height = geodesic(coords_ul, coords_dl).km
    img_diagonal = geodesic(coords_ul, coords_dr).km

    # only applicable if the images form a recangle overall
    land_area = land_width * land_height

    # only applicable if the images forming a rectangle do not overlap
    img_area = land_area / len(data)

    pretty('Area Diagonal Distance:', img_diagonal, ' Km',
                    '\nWidth =', land_width, 'Km',
                    '\nHeight =', land_height, 'Km',
                    '\nLand area = ', land_area, 'Km^2',
                    '\nArea covered by each image =', img_area, 'Km^2',
                    header='[INFO] Analyzing downloaded data. Attention!')

    return data_min, data_max


def meters2geo(
    center: Tuple[float, float], img_size: Tuple[float, float], epsg
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Function: Converts the center point and image size from meters
        to geographic coordinates and returns the top left and bottom right coordinates.
    Parameters:
        - center (tuple): A tuple containing the center point coordinates in meters (x, y).
        - img_size (tuple): A tuple containing the width and height of the image in meters.
    Returns:
        - tl (tuple): A tuple containing the top left coordinates in latitude and longitude.
        - br (tuple): A tuple containing the bottom right coordinates in latitude and longitude.
    Processing Logic:
        - Convert center point from meters to UTM coordinates.
        - Calculate the bottom right and top left coordinates in UTM.
        - Convert the UTM coordinates to geographic coordinates.
        - Return the top left and bottom right coordinates.
    """

    # Compute half the width and height of the image
    img_w_m, img_h_m = np.array(img_size) / 2
    # Convert center point from geographic to UTM coordinates
    cxm, cym = geo2utm(center[0], center[1], epsg)
    # Calculate top left and bottom right UTM coordinates
    brm = (cxm + img_w_m, cym - img_h_m)
    tlm = (cxm - img_w_m, cym + img_h_m)

    # Convert UTM coordinates back to geographic coordinates
    tl = utm2geo(tlm[0], tlm[1], epsg)
    br = utm2geo(brm[0], brm[1], epsg)

    return tl, br


@tf.function
def geodist_loss_params(data_obj):
    """
        Loss funcion to apply haversine distance difference between
        unnormalized inputs.
        todo: update with scaler
    """
    def geodist_loss(y_pred, y_true):
        y_pred = data_obj.scaler.inverse_transform(y_pred)
        y_true = data_obj.scaler.inverse_transform(y_true)

        lat1, lon1 = tf.unstack(y_pred, axis=-1)
        lat2, lon2 = tf.unstack(y_true, axis=-1)

        return haversine_distance((lat1, lon1, lat2, lon2))
    return geodist_loss


@tf.function
def haversine_distance(coords):
    """
    Calculates the haversine distance between two coordinates in meters.
    Parameters:
        - coords (list): List of four coordinates in decimal degrees [lat1, lon1, lat2, lon2].
    Returns:
        - distance (float): Haversine distance between the two coordinates in meters.
    Processing Logic:
        - Convert coordinates to radians.
        - Calculate differences in latitude and longitude.
        - Calculate haversine formula.
        - Convert distance to meters.
    """
    coords = [x * np.pi / 180.0 for x in coords]
    dlat = coords[2] - coords[0]
    dlon = coords[3] - coords[1]

    a = tf.math.sin(dlat / 2) ** 2  +\
        tf.math.cos(coords[0]) * tf.math.cos(coords[2]) * tf.math.sin(dlon / 2) ** 2
    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1 - a))

    distance = consts.EARTH_RADIUS * c
    return distance * 1000


def geo2utm(lat: float, lon: float, epsg: str) -> Tuple[float, float]:
    """
    Converts geographic coordinates to UTM coordinates.
    Parameters:
        - lat (float): Latitude in decimal degrees.
        - lon (float): Longitude in decimal degrees.
        - epsg (str): EPSG code for desired UTM zone.
            Defaults to EPSG code for current UTM zone.
    Returns:
        - x (float): UTM easting coordinate.
        - y (float): UTM northing coordinate.
    Processing Logic:
        - Uses pyproj library for coordinate transformation.
        - EPSG code is used to determine UTM zone.
        - Coordinates are returned in meters.
    """

    x, y = pyproj.Transformer.from_crs(
        "EPSG:4326", epsg, always_xy=True
        ).transform(lon, lat)
    return x, y


def utm2geo(x: float, y: float, epsg: str) -> Tuple[float, float]:
    """
    Converts UTM coordinates to geographic coordinates.
    Parameters:
        - x (float): UTM x-coordinate.
        - y (float): UTM y-coordinate.
        - epsg (str): EPSG code for UTM zone. Defaults to consts.ARGS.utm.
    Returns:
        - Tuple[float, float]: Geographic coordinates (latitude, longitude).
    Processing Logic:
        - Uses pyproj library for coordinate transformation.
        - EPSG code is required for accurate conversion.
        - Coordinates are returned in (latitude, longitude) format.
    """
    lon, lat = pyproj.Transformer.from_crs(
        epsg, "EPSG:4326", always_xy=True
        ).transform(x, y)
    return lat, lon

def calc_bbox_m(center_coords, bbox_m):
    """
    Calculates the bounding box coordinates from a given center point and dimensions.
    Parameters:
        - center_coords (tuple[float, float]): Latitude and longitude of the center point.
        - bbox_m (tuple[float, float]): X, Y Widths of the bounding box in meters.
    Returns:
        - tuple[tuple[float, float], tuple[float, float]]:
            Coordinates of the top-left and bottom-right corners of the bounding box.
    Example:
        - calc_bbox_m((37.7749, -122.4194), 2000, 1000)
                -> ((37.7840, -122.4293), (37.7658, -122.4095))
    """
    center_point = Point(center_coords)

    # Calculate half of the width and height in degrees
    half_x = bbox_m[0] / 2
    half_y = bbox_m[1] / 2
    # Top-left corner (north-west)
    top_left = geodesic(meters=half_y).destination(center_point, 0)  # North
    top_left = geodesic(meters=half_x).destination(top_left, 270)  # West

    # Bottom-right corner (south-east)
    bottom_right = geodesic(meters=half_y).destination(center_point, 180)  # South
    bottom_right = geodesic(meters=half_x).destination(bottom_right, 90)  # East

    return (top_left.latitude, top_left.longitude), (bottom_right.latitude, bottom_right.longitude)

def get_map_dim_m(fov_d, agl_m, aspect_ratio):
    # agl_m = agl_f * 0.3048 # convert feet to meters
    # Calculate diagonal in meters in image using fov and ar
    d_m = 2 * agl_m * np.tan(np.radians(fov_d/2))

    # Calculate width and height in meters from the diagonal
    return (d_m * np.sin(np.arctan(aspect_ratio)),
            d_m * np.cos(np.arctan(aspect_ratio)),
            agl_m)


def get_utm_epsg(coords):
    lat, lon = coords
    # Determine UTM zone number
    zone_number = int((lon + 180) / 6) + 1
    # Determine if it is in the Northern or Southern hemisphere
    if lat >= 0:
        epsg_code = f"EPSG:326{zone_number:02d}"  # Northern hemisphere
    else:
        epsg_code = f"EPSG:327{zone_number:02d}"  # Southern hemisphere
    return epsg_code
