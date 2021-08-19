# ESPNetv2_pytorch

## Set miniconda environment
```
set_conda
conda activate <env-name> #start up
conda deactivate <env-name> #close
```

## Execution ESPNetv2
delete the cached data file (~/ESPNetv2-master/segmentation/city.p)
```
conda activate opencv_build #start up environment
CUDA_VISIBLE_DEVICES=0 python3 main_hao.py --s 2.0 #start to train
CUDA_VISIBLE_DEVICES=0 python3 prediction_ros.py --s 2.0 --pretrained ./model_best.pth #evaluate the module
```
