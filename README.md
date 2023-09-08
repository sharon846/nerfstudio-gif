# nerfstudio-gif

We present an extension of nerfstudio amazing work, in gif completion.
Record the video which you want to make gif, train nerf on it and render the new circular video.

## Installation:
0. run ```apt update```.
1. Install colmap, like detailed [here](https://colmap.github.io/install.html#linux). <br/>
   After cloning the colmap git, add to CMakeLists.txt, under "Dependency configuration" the macro ```set(CMAKE_CUDA_ARCHITECTURES "native")```, to auto configure your GPU.
2. install ffmpeg
3. Run the following:
```
pip install --upgrade pip setuptools
git clone https://github.com/sharon846/nerfstudio.git
cd nerfstudio
pip install -e .
```
Althernative option, is to install the original nerfstudio project, but make sure to sync the colmap arguments in proccess_data/colmap_util.py

### Installing in DLC
The colmap installtion takes a long time. Therefore, we will prefer to install it into container. You can start with an existing container by specifing it or creat it from scratch (but you will probably have to install more libraries). We recommend to run it in front (not in background).
Run: ```srun -G 2 --container-image=[existing container] --container-save=[path to save the new container] --pty /bin/bash```.
After the container loaded, make all the installes as above.

## Process video
Put the wanted video in directory with your model name. For example, Room/video.mp4. This folder will be used to the process & training outputs along with the final rendering.

In that directory, run the following commands:
```
1. rm -rf Images colmap
2. mkdir Images
3. ffmpeg -i video.mp4 -vf fps=30 Images/output_%04d.png
4. ns-process-data images --data Images --output-dir colmap
```
You may replace the fps with the fps of your camera.

After colmap successfully proccessed <b>all</b> the frames, run our script using ```python organize_json.py```. <br/>
Make sure to set colmap_dir to the colmap directory created for your data. 

## Training nerf:
Run ```ns-train [model] --vis viewer --pipeline.datamanager.train-num-rays-per-batch 4096 --data colmap```. <br/>
Notes:
1. For our data, we used nerfacto-huge, but the full models list available [here](https://docs.nerf.studio/en/latest/nerfology/methods/).
2. When having more than 1 GPU, you may use them by specifing --machine.num-devices [gpus_num]. If you have only one, you can drop this part.

## Gif rendering:
Run ```ns-render-gif.sh [your_model_name]```. This will do all the job for you :) <br/>
Make sure to edit ```data_dir=[Parent_of_your_model_directory]/$1```.

Note: the rendered gif has blue (old) and green (new) border for visualization. You may block the line that calls ```pad_images.py```.

## Output structure
At the end of the process, the following structure will be in your model folder
```
.
├── ...
├── render                    # stores the frames of your gif
├── plots                     # stores the 3d plotting of the camera track (pos + lookat) in your gif
├── outputs                   # stores the nerf output along with final videos.
│   ├── colmap                # The folder of the nerf train itself. Also stores the generated camera path
│   ├── video.mp4             # The new rendered gif
│   ├── camera.mp4            # The animation of the camera track (pos + lookat) in your gif
│   └── points_viewer.html    # 3d plotting of the original frames, the frames deleted by out loop_detect algorithm, the outlier frames and the new generated ones
└── ...
```

## Example - Room
(https://drive.google.com/file/d/1oyYOyAqeX4gMJdbxmeMeZadOHLYcCwfE/view?usp=drive_link)[original gif]
https://github.com/sharon846/nerfstudio-gif/assets/62396923/410e0d08-eb4b-492a-bece-30cd67bde5ea <br/><br/>

new gif + camera tracking: <br/>
![points](https://github.com/sharon846/nerfstudio-gif/blob/master/dmo.gif)

## Algorithm and Data
Our paper including the theory and algoritmic ideas behind this project is available [here](https://github.com/sharon846/nerfstudio-gif/blob/master/paper.pdf) <br/>
Our data is available [here](https://drive.google.com/drive/folders/1Ob5fAqlgcwI0g1AiODrqTvwFuIHd3IJp?usp=drive_link)

