# terrAInav Sim: An Open-Source Simulation of UAV Aerial Imaging from Map-Based Data

Capturing real-world aerial images for vision-based navigation (VBN) is challenging due to limited availability and conditions that make it nearly impossible to access all desired images from any location. The complexity increases when multiple locations are involved. State-of-the-art solutions, such as deploying UAVs (unmanned aerial vehicles) for aerial imaging or relying on existing research databases, come with significant limitations. TerrAInav Sim offers a compelling alternative by simulating a UAV to capture bird’s-eye view map-based images at zero yaw with real-world visible-band specifications. This open-source tool allows users to specify the bounding box (top-left and bottom-right) coordinates of any region on a map. Without the need to physically fly a drone, the virtual Python UAV performs a raster search to capture images. Users can define parameters such as the flight altitude, aspect ratio, diagonal field of view of the camera, and the overlap between consecutive images. TerrAInav Sim’s capabilities range from capturing a few low-altitude images for basic applications to generating extensive datasets of entire cities for complex tasks like deep learning. This versatility makes TerrAInav a valuable tool for not only VBN but also other applications, including environmental monitoring, construction, and city management. The open-source nature of the tool also allows for the extension of the raster search to other missions. A dataset of Memphis, TN, has been provided along with this simulator. A supplementary dataset is also provided, which includes data from a 3D world generation package for comparison.

# Getting Started:
## Requirements:

- Python 3.10
- pip
- conda (recommended)
- tested on both windows and linux

## Initiate and activate the environment:

<details>
  <summary>Setup the environment (Highly recommended)</summary>

    conda create -n "terrainav-sim" python=3.10
    conda activate terrainav-sim

</details>


<details>
  <summary>Enable GPU access (Not necessary)</summary>

Other versions might be compatible at your own risk. Check compatibility here: https://www.tensorflow.org/install/source#gpu

    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1

</details>


### Install dependencies:

    pip install -r requirements.txt

## Access and use Memphis Agricenter dataset:

Check src/notebooks folder.

## Download the Data:

### A single coordinate

    python -m src.download_single

<details>
  <summary>To specify more features:</summary>

You can do either:

    python -m src.download_single --coords /path/to/file --aspect_ratio <X> <Y> --fov <degrees> --data_dir /path/to/dataset

The file should have one line including the following data:

    <latitude> <longitude> <AGL(f)>

Check dataset/sample_coords.txt as an example.

Or:

    python -m src.download_single --coords "<Latitude>_<Longitude>_<AGL(feet)>"

for more configuration parameters checkout src/utils/config.py or type:

    python -m src.download_raster --help

- **For example:**

        python -m src.download_single --coords "dataset/sample_coords.txt" --aspect_ratio 4 3 --fov 78.8 --data_dir dataset/

    Note: the aspect ratio and fov are from DJI Mavic and are set to the above values by default.

- Or:

        python -m src.download_single --coords "35.22_-90.07_400"

</details>

### A list of coordinates

    python -m src.download_from_list --coords /path/to/file

### Raster Mission
    python -m src.download_raster --coords "<TopLeftLatitude>_<TopLeftLongitude>_<TopLeftLatitude>_<TopLeftLongitude>_<AGL(feet)>"

- **For example:**

        python -m src.download_raster --coords "35.22_-90.07_35.06_-89.73_400"



## Google Colab Repository for local edits:

https://colab.research.google.com/drive/1Huaq96ssyPMy7Xx1IVcUQaoPZu0Idfhk?usp=sharing

<details>
  <summary>How to run colab on local server machine:</summary>

1. SSH to the remote directory and forward the port such as:

```
ssh -L localhost:8888:localhost:8888 [username]@[hostname or IP address]
```

2. Run the following in the remote terminal:
```
jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
```
3. The previous step will give you two URLs in result. Copy either. For example:
```
http://localhost:8888/?token=0f96a96950ca8aa79c52fb1fa5758e648b5052cd91417dd8
```
or
```
http://127.0.0.1:8888/?token=0f96a96950ca8aa79c52fb1fa5758e648b5052cd91417dd8
```
4. On the bar above select the arrow next to the connect button and choose "connect to a local runtime".
5. A popup window will be shown, paste the copied URL in the input section.
6. Press "Connect" and voila.
7. If you are using a conda environment for your packages, you might need the following steps. On the remote server, install ipykernel:
```
conda install ipykernel
```
1. Then, register the Conda environment as a Jupyter/Colab kernel (Replace <environment_name> with the name of your Conda environment):
```
!python -m ipykernel install --user --name=<environment_name>
```
1. After installing and registering the kernel, you can switch to it from within your Colab notebook interface by selecting it from the kernel dropdown menu (click on "Runtime" > "Change runtime type" > select your Conda environment).
2.  Always restart the Colab runtime after setting up a custom kernel or installing packages to ensure the changes take effect. Click on "Runtime" in the menu and select "Restart runtime...".
</details>

## Licensing Information

- The source code and the data in this repository is licensed under the GPL-3.0 License (`LICENSE`).
- The paper is licensed under [CC Y 4.0 License](https://creativecommons.org/licenses/by/4.0/).

## Paper Link
The paper can be downloaded from [this link](https://www.mdpi.com/2072-4292/17/8/1454). Please cite our paper when using any of our content:

```
@Article{rs17081454,
AUTHOR = {Dajkhosh, Seyedeh Parisa and Le, Peter M. and Furxhi, Orges and Jacobs, Eddie L.},
TITLE = {TerrAInav Sim: An Open-Source Simulation of UAV Aerial Imaging from Map-Based Data},
JOURNAL = {Remote Sensing},
VOLUME = {17},
YEAR = {2025},
NUMBER = {8},
ARTICLE-NUMBER = {1454},
URL = {https://www.mdpi.com/2072-4292/17/8/1454},
ISSN = {2072-4292},
DOI = {10.3390/rs17081454}
}
```
