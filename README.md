# Deep_Lane_Segmentation
### KMU HCI Lab
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
│
└── result
     ├── Unet_weight
     └── SHG_weight
```

## Inference
#### Download Weight File
- [Weight File](https://drive.google.com/drive/folders/18q5HJHThWr_uRyEm6_CkWK74C5ufr2WQ?usp=sharing) Download SHG_weight, Unet_weight
> mkdir result && cd result && tar -zxvf SHG_weight.tar.gz && tar -zxvf Unet_weight.tar.gz 

#### Stacked Hourglass Network
> python3 inference.py --netType stackedHGB --resume ../result/SHG_weight --nStack 7
#### Unet
> python3 inference_unet.py --netType Unet --resume ../result/Unet_weight

## Inference Video
> KMU Self-Driving-Studio

![av_gif](./gif/Driving_Studio_Output.gif)
