# D_Lane_Segmentation
Deep Lane Segmentation using SHG Module

-----
Folders
-----
| Folder     | Description                  |
|------------|------------------------------|
| models      | network architectures and parameters |
| datasets   | define dataloader interfaces |
| criterions | define criterion interfaces  |
| util       | visulization utilities       |

-----
Directory
-----
```
D_Lane_Segmentation
├── src
│    ├── models
│    ├── input_video
│    ├── datasets   
│    ├── output_video
│    └── util
|
└── result
      ├── Unet_weight
      └── SHG_weight
```

> 2020.07.15
> * infence code create

## Inference
#### Stacked Hourglass Network
> python3 inference.py --netType stackedHGB --nStack 7
#### Unet
> python3 inference_unet.py --netType Unet

## Inference Video
> KMU Self-Driving-Studio

![av_gif](./gif/Driving_Studio_Output.gif)
