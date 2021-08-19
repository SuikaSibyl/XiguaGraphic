# XiguaGraphic

## Warnings
The program is based on "Introduction to 3D Game Programming with DirectX12", and provided support for only DX12 Raster API.
Sadly, although after struggling for writing complex DX12 codes I finally implemented some great stuff, I gradually to realize
the structure of the program is too bad. Basically speaking, it's too bad in encapsuling and decoupling, so I would not want to 
rehash this mess anylonger. I'm now working on something better arranged and designed, and the new program is going to get publiced later, 
maybe 2021/9 I guess.

## Environment
Windows + VS2019 + Cuda10.2 + Qt 5.14

Please Donwload Important Resources form BaiduNetdisk:

webpage：https://pan.baidu.com/s/19CuW0dWqvodTyRh1wOQJcw

code：j0p1 


## Content
DX12 + Cuda implementation of some rendering algorithms.
For rasterization part, I implemented:
- Shadowmap & PCF soft shadows
- The split sum approximation with IBL

  ![image](https://github.com/SuikaSibyl/XiguaGraphic/blob/main/figs/SplitSum.png)
- PRT with multiple polygon lights, also with Translucent Material

  It's based on paper "Analytic Spherical Harmonic Gradients for Real-Time Rendering with Many Polygonal Area Lights"
  
  ![image](https://github.com/SuikaSibyl/XiguaGraphic/blob/main/figs/PRT.png)
  ![image](https://github.com/SuikaSibyl/XiguaGraphic/blob/main/figs/MultiLights.png)
  ![image](https://github.com/SuikaSibyl/XiguaGraphic/blob/main/figs/Transparent.png)
  ![image](https://github.com/SuikaSibyl/XiguaGraphic/blob/main/figs/PRTPipeline.png)
- SSAO / SSDO post processing

  ![image](https://github.com/SuikaSibyl/XiguaGraphic/blob/main/figs/SSDO.png)
Also, I implemented basic Path Tracer using CUDA.
  ![image](https://github.com/SuikaSibyl/XiguaGraphic/blob/main/figs/PathTracer.png)
