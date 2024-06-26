## Pytorch Implementation of PointNet and PointNet++ Trained on KITTI Point Cloud Semantic Segmentation dataset

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.
Links for Official Code:
[Official PointNet](https://github.com/charlesq34/pointnet) and [Official PointNet++](https://github.com/charlesq34/pointnet2)

# Install
```bash
conda create --name tf python=3.6
conda activate tf
conda install -c pytorch pytorch torchvision
conda install -c open3d-admin open3d=0.9.0.0
pip install pyyaml matplotlib tqdm h5py redis numpy pandas opencv-python==4.2.0.32
```

# Run demo
```bash
export KITTI_ROOT=PATH/odometry/dataset
# windows: $env:KITTI_ROOT = 'PATH/odometry/dataset'
python pcdvis.py
```