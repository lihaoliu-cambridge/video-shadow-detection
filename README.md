# Pytorch Lightning Implemented TVSD

This repo contains the Pytorch Lightning version ["Triple-cooperative Video Shadow Detection, CVPR'21"](https://arxiv.org/pdf/2103.06533.pdf) based on Zhihao Chen's original [Pytorch implemented TVSD](https://github.com/eraserNut/ViSha) written by Lihao Liu.


Instead of writing a lot of code for the training logic (such as data parallel on multiple GPUs, map tensors to GPU using .cuda() function, visualization using tensorboard, and so on), we will use the high-level PyTorch framework PyTorch Lightning to manage the training and testing logic. Moreover, we simplify the data loaders in the original repo to enhance the readability for video data preprocessing.


## Requirement

cuda==11.1   
cudnn==8.0  

torch==1.9.0  
pytorch-lightning==1.5.10  
tensorboard==2.7.0  
tensorboardX==2.5  
apex==0.1   (download from https://github.com/NVIDIA/apex)  


## Usage

1. Clone the repository:

```shell
git clone https://github.com/lihaoliu-cambridge/video-shadow-detection.git
cd video-shadow-detection
```
   
2. Download and unzip [Visha dataset](https://erasernut.github.io/ViSha.html), and put the unzipped Visha directory into the dataset directory:
   
```shell
./dataset/Visha
```
   
3. Modify the configurations for data loader, model architecture, and training logic in:
      
```shell
./config/visha_tvsd_config.yaml
```

4. Download the pre-trained weight for the model backbone [resnext_101_32x4d.pth](https://drive.google.com/file/d/1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ/view), and modify the path to this pre-trained weight in:

```shell
./model/resnext_modify/config.py
resnext_101_32_path = '[Your Project Directory]/backbone_pth/resnext_101_32x4d.pth'
```
   
5. Train the model:
 
```shell
python train.py
```

6. Run the tensorboard monitor and open the tensorboard link: (http://127.0.0.1:6006/) in your browser:

```shell
tensorboard --port=6006  --logdir=[Your Project Directory]/output/tensorboard/tvsd_visha
```
<img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/vsd_loss.jpg" width="480"/>  
<img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/vsd_visualization.png" width="960"/>  

7. After training, modify the generated checkpoint file path in test.py file, and then you can test the trained model:
 
```shell
python test.py  
```


## Citation

```
@inproceedings{chen2021triple,
   title={Triple-cooperative video shadow detection},
   author={Chen, Zhihao and Wan, Liang and Zhu, Lei and Shen, Jia and Fu, Huazhu and Liu, Wennan and Qin, Jing},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages={2715--2724},
   year={2021}
}
```
