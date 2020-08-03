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
│    ├── App
│    |   ├── screens
│    |       ├── Admin
│            │   └── screens
│            │       ├── Reports
│            │       └── Users
│            └── Course
│                └── screens
│                    └── Assignments
└── result
      └── Unet_weight
      └── SHG_weight
```

> 2020.07.15
> * infence code create

## Inference
> python3 inference.py --netType stackedHGB --GPUs 0 --LR 0.001 --nStack 7 --batchSize 1

## Inference Video
> KMU Self-Driving-Studio

![av_gif](./gif/Driving_Studio_Output.gif)
