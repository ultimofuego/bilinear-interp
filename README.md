# bilinear-interp
Bilinear interpolation 

In mathematics, bilinear interpolation is an extension of linear interpolation for interpolating functions of two variables (e.g., x and y) on a rectilinear 2D grid.

Bilinear interpolation is performed using linear interpolation first in one direction, and then again in the other direction. Although each step is linear in the sampled values and in the position, the interpolation as a whole is not linear but rather quadratic in the sample location.

Bilinear interpolation is one of the basic resampling techniques in computer vision and image processing, where it is also called bilinear filtering or bilinear texture mapping.

## Prerequisites
1. Microsoft visual studio 19
2. Nvidia GPU (cuda SUPPORT)
3. CUDA Toolkit 11
4. EasyBMP
## Build and Run
1. Make new CUDA-project.
2. Include in the project "bilinear-interp.cu".
## System configuration
| Name  | Values  |
|-------|---------|
| CPU  | Intel® Pentium® G3430 (2x3.30 GHz) |
| RAM  | 4 GB DDR3 |
| GPU  | GeForce GTX 750 Ti 2GB |
| OS   | Windows 10 64-bit  |
## Results
<img src="https://github.com/ultimofuego/bilinear-interp/blob/master/cat250x188.bmp" /> |
------------ |
Input 250 x 188

<img src="https://github.com/ultimofuego/bilinear-interp/blob/master/CatGPUout.bmp" /> | <img src="https://github.com/ultimofuego/bilinear-interp/blob/master/CatCPUout.bmp" />
------------ | ------------- 
Output GPU 500 x 376 | Output CPU 500 x 376

Average results after 100 times of runs.

|    Input size  |   Output size |          CPU        |         GPU       | Acceleration |
|-------------|-|--------------------|-------------------|--------------|
| 500x375   | 1000x750 |18 ms               | 6.5 ms            |    2.77      |
| 1000x750   | 2000x1500 |57 ms               | 22 ms            |    2.59      |
| 4000x3000   | 8000x6000 |808 ms              | 339 ms             |    2.38      |
