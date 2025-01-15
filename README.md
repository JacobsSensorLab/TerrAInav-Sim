# terrAInav Sim: An Open-Source Simulation of UAV Aerial Imaging from Satellite Data

Capturing real-world aerial images is challenging due to limited availability and conditions that make it nearly impossible to access all desired images from any location. The complexity increases when multiple locations are involved. Traditional solutions, such as flying a UAV (Unmanned Aerial Vehicle) to take pictures or using existing research databases, have significant limitations. terrAInav Sim offers a compelling alternative by simulating a UAV to capture aerial images with real-world visible-band specifications. This open-source tool allows users to specify the top-left and bottom-right coordinates of any region on a map. Without the need to physically fly a drone, the virtual UAV performs a raster search to capture satellite images using the Google Maps API. Users can define parameters such as flight altitude in meters, aspect ratio and diagonal field of view of the camera, and the overlap between consecutive images. terrAInav Sim's capabilities range from capturing a few low-altitude images for basic applications to generating extensive datasets of entire cities for complex tasks like deep learning. This versatility makes terrAInav a valuable tool for various applications in environmental monitoring, construction, and city management. The open-source nature of the tool also allows for extending the raster search to any other desired path. It is important to note that while the search can theoretically be conducted worldwide, the most accurate results are achieved if users stay within the same or nearby UTM (Universal Transverse Mercator) zone. Therefore, specifying the zone for the program is recommended. Using terrAInav Sim, a few examples of Memphis, TN are also provided.

## Requirements:

- Python 3.10
- pip
- conda (recommended)
- tested on both windows and linux

## Initiate and activate the environment:

<details>
  <summary>Setup the environment (Highly recommended)</summary>

    conda create -n "terrAInav-sim" python=3.10
    conda activate terrAInav-sim

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

## Create a GoogleMap API key:
**This step is for generating your own data.**
Check how you can create one here: https://developers.google.com/maps/documentation/javascript/get-api-key

Then make a python file in the src folder called: hidden_file.py and add the following line to it:

    api_key='<Your API Key>'

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

    python -m src.main --help

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

## License

This project is licensed under the [CC Y 4.0 License](https://creativecommons.org/licenses/by/4.0/).

## Paper Link
The paper can be downloaded from [this link](https://www.arxiv.org/abs/2410.02003). Please cite our paper when using the dataset:

```
@misc{dajkhosh2024terrAInavsimopensourcesimulation,
      title={terrAInav Sim: An Open-Source Simulation of UAV Aerial Imaging from Satellite Data},
      author={S. Parisa Dajkhosh and Peter M. Le and Orges Furxhi and Eddie L. Jacobs},
      year={2024},
      eprint={2410.02003},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.02003},
}
```
