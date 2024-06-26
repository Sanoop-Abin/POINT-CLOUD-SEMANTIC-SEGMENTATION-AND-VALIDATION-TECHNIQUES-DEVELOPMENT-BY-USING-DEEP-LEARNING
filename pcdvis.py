import open3d
import argparse
import os
import time
import json
import h5py
import datetime
import cv2
import yaml
import colorsys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt
import my_log as log
import matplotlib.pyplot as plt
from model.pointnet import PointNetSeg, feature_transform_reguliarzer
from model.pointnet2 import PointNet2SemSeg
from model.utils import load_pointnet

from pcdseg import parse_args
from data_utils.SemKITTI_Loader import pcd_normalize
from data_utils.kitti_utils import Semantic_KITTI_Utils

#KITTI_ROOT = os.environ['KITTI_ROOT']
KITTI_ROOT = "D:\Dataset\data_odometry_color\dataset"
class Window_Manager():
    def __init__(self):
        self.param = open3d.io.read_pinhole_camera_parameters('config/ego_view.json')
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=800, height=800, left=100)
        self.vis.register_key_callback(32, lambda vis: exit())
        self.vis.get_render_option().load_from_json('config/render_option.json')
        self.pcd = open3d.geometry.PointCloud()
    
    def update(self, pts_3d, colors):
        self.pcd.points = open3d.utility.Vector3dVector(pts_3d)
        self.pcd.colors = open3d.utility.Vector3dVector(colors/255)
        self.vis.remove_geometry(self.pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.param)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def capture_screen(self,fn):
        self.vis.capture_screen_image(fn, False)

def export_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    font = cv2.FONT_HERSHEY_SIMPLEX
    out = cv2.VideoWriter('experiment/pn_compare.avi',fourcc, 15.0, (int(1600*0.8),int(740*0.8)))

    #mkdir('experiment/imgs/%s/'%(args.model_name))
    #vis_handle.capture_screen('experiment/imgs/%s/%d_3d.png'%(args.model_name,i))
    #cv2.imwrite('experiment/imgs/%s/%d_sem.png'%(args.model_name, i), img_semantic)
    
    for index in range(100, 320):
        pn_3d = cv2.imread('experiment/imgs/pointnet/%d_3d.png' % (index))
        pn_sem = cv2.imread('experiment/imgs/pointnet/%d_sem.png' % (index))
        pn2_3d = cv2.imread('experiment/imgs/pointnet2/%d_3d.png' % (index))
        pn2_sem = cv2.imread('experiment/imgs/pointnet2/%d_sem.png' % (index))

        pn_3d = pn_3d[160:650]
        pn2_3d = pn2_3d[160:650]

        pn_sem = cv2.resize(pn_sem, (800, 250))
        pn2_sem = cv2.resize(pn2_sem, (800, 250))

        pn = np.vstack((pn_3d, pn_sem))
        pn2 = np.vstack((pn2_3d, pn2_sem))

        cv2.putText(pn, 'PointNet', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn, 'PointNet', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn2, 'PointNet2', (20, 100), font,1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pn2, 'PointNet2', (20, 520), font,1, (255, 255, 255), 2, cv2.LINE_AA)

        merge = np.hstack((pn, pn2))
        class_names = ['unlabelled', 'vehicle', 'human', 'ground', 'structure', 'nature']
        colors = [[255, 255, 255],[245, 150, 100],[30, 30, 255],[255, 0, 255],[0, 200, 255],[0, 175, 0]]
        for i,(name,c) in enumerate(zip(class_names, colors)):
            cv2.putText(merge, name, (200 + i * 200, 50), font,1, [c[2],c[1],c[0]], 2, cv2.LINE_AA)

        cv2.line(merge,(0,70),(1600,70),(255,255,255),2)
        cv2.line(merge,(800,70),(800,1300),(255,255,255),2)

        merge = cv2.resize(merge,(0,0),fx=0.8,fy=0.8)
        # cv2.imshow('merge', merge)
        # if 32 == waitKey(1):
        #     break
        out.write(merge)

        print(index)
    out.release()

def vis(args):
    part = '02'
    args.subset ='inview'
    args.model_name = 'pointnet'

    kitti_utils = Semantic_KITTI_Utils(KITTI_ROOT, subset=args.subset)
    #print(kitti_utils.num_classes)
    vis_handle = Window_Manager()
    if args.model_name == 'pointnet':
        args.pretrain = 'checkpoints/pointnet-inview-0.59407-0008.pth'
    else:
        args.pretrain = 'checkpoints/pointnet2-inview-0.55884-0001.pth'

    model = load_pointnet(args.model_name, kitti_utils.num_classes, args.pretrain)

    for index in range(0, kitti_utils.get_max_index(part)):
        point_cloud, label = kitti_utils.get(part, index, load_image=True)
        
        # resample point cloud
        length = point_cloud.shape[0]
        npoints = 25000
        choice = np.random.choice(length, npoints, replace=True)
        point_cloud = point_cloud[choice]
        label = label[choice]

        pts_3d = point_cloud[:,:3]
        pcd = pcd_normalize(point_cloud)

        with log.Tick():
            points = torch.from_numpy(pcd).unsqueeze(0).transpose(2, 1).cuda()
            print("11111111111111111111111111111111111111111111111111111")
            print(points.shape)
            
            with torch.no_grad():
                if args.model_name == 'pointnet':
                    logits, _ = model(points)
                else:
                    logits = model(points)
                pred = logits[0].argmax(-1).cpu().numpy()

            print(index, pred.shape, end='')
            #plt.plot(pred, ".")
           # plt.show()
        pts_2d = kitti_utils.project_3d_to_2d(pts_3d)
        #pts_2d = kitti_utils.torch_project_3d_to_2d(pts_3d)

        vis_handle.update(pts_3d, kitti_utils.colors[pred])
        sem_img = kitti_utils.draw_2d_points(pts_2d, kitti_utils.colors_bgr[pred])

        cv2.imshow('semantic', sem_img)
        cv2.imshow('frame', cv2.cvtColor(kitti_utils.frame,cv2.COLOR_BGR2RGB))
        if 32 == cv2.waitKey(1):
            break

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    vis(args)
   
