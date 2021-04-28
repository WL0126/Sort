"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True) # 使用lap.lapjv实现线性分配
    # 返回分配花费(cost)和两个数组x,y;如果成本矩阵cost_matrix形状为NxM,则x为大小为N的数组,指定将行分配给哪一列,y为大小为M的数组,指定每列分配给哪行。
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt): # IOU的实现----交并比
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox): # 将BBox从[x1,y1,x2,y2]格式变为[x,y,s,r]格式
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None): # 将BBox从[x,y,s,r]变为[x1,y1,x2,y2]的格式 -- (x1,y1)左上角,(x2,y2)右下角
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):  # 卡尔曼滤波器  自身预测+测量结果的反馈
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    # kf.F 状态转移矩阵 -- 7X7
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    # kf.H 测量函数 -- 4X7
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
    # kf.R 状态不确定性 -- 7X7
    self.kf.R[2:,2:] *= 10.
    # kf.P 不确定性协方差 -- 7X7  (初始的速度分量也具有高的不确定性）
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    # kf.Q 过程不确定性 -- 7X7
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox) # x保存BBox坐标和ID 前4位保存BBox坐标
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3): # 将物体检测的BBox和卡尔曼滤波器预测的BBox匹配
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):# 刚开始时 跟踪BBox为空,trks=[]即len(trackers)==0
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
        # 因为没有trks所以match为[],因此新增物体的矩阵就是检测的矩阵,没有离开画面的物体，所以第三个也为[]

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix) # 通过匈牙利算法匹配卡尔曼滤波器预测的BBox与物体检测BBox以[[d,t]...]的二维矩阵保存到match_indices
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections): # 遍历物体检测BBox集合，每个BBox标识为d
    if(d not in matched_indices[:,0]): # 没有匹配上的物体检测BBox放入unmatched_detections列表;表示有新的物体进入画面了，后面要新增跟踪器追踪新物体
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers): # 遍历卡尔曼滤波器预测的BBox集合，每个BBox标识为t
    if(t not in matched_indices[:,1]): # 没有匹配上的卡尔曼滤波器预测的BBox放入unmatched_trackers列表;表示之前跟踪的物体离开画面了，后面要删除对应的跟踪器
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices: # 遍历matched indices,将IOU值小于iou_threshold的匹配结果分别放入unmatched_detections与unmatched_trackers列表中。
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2)) # 匹配上的卡尔曼滤波器预测的BBox与物体检测BBox以[[d,t]...]的形式放入matches矩阵
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers) # 返回跟踪成功的物体矩阵,新增物体矩阵,离开画面的物体矩阵


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))): # 输入是当前帧中所有物体的检测BBox集合，包括物体的score;输出是当前帧的物体跟踪BBox集合，包括物体跟踪的ID
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5)) # 根据当前所有的卡尔曼跟踪器的个数（等于上一帧中被跟踪的物体个数）创建二维矩阵trks。行号为卡尔曼跟踪器表示，列向量为跟踪BBox与物体跟踪ID -- 4+1
                                             # 刚开始检测时只有 检测BBox即det 没有 跟踪BBox即trks
    to_del = []
    ret = []
    for t, trk in enumerate(trks): # 循环遍历卡尔曼跟踪器列表
      pos = self.trackers[t].predict()[0] # 用卡尔曼跟踪器t产生对于物体的当前帧中预测的BBox
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    # np.ma.masked_invalid用于将数组trks中的无效值（Nans or infs）设置为mask(--)
    # np.ma.compress_rows抑制二维数组中包含屏蔽值的整行
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks)) # trks中存放了上一帧中被跟踪的所有物体在当前帧中预测的BBox
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)
                                                           # 将物体检测的BBox与卡尔曼滤波器预测的跟踪BBox匹配
                                                           # 获得跟踪成功的物体矩阵，新增物体的矩阵，离开画面的物体矩阵
    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        a = 1
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers): # reversed用于列表中数据的反转 [1,2,3,4]--[4,3,2,1]
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i) # 离开画面/跟踪失败的物体从物体的卡尔曼跟踪器列表中删除
    if(len(ret)>0):
      # np.concatenate()完成对多个数组的拼接
      return np.concatenate(ret) # 返回当前画面中所有被跟踪物体的BBox与ID
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""    # 用于命令行选项和参数解析。
    parser = argparse.ArgumentParser(description='SORT demo') # 创建一个解析对象
    # 向该对象中添加一个你要关注的命令行参数和选项
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    # 进行解析
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args() # 参数解析后
  display = args.display # False
  phase = args.phase # train

  total_time = 0.0 # 总时间
  total_frames = 0 # 总帧数
  colours = np.random.rand(32, 3) #used only for display // BBox颜色种类

  if(display):# 如果要显示跟踪结果到屏幕上，2D绘图初始化
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion() # 使matplotlib的显示模式由默认的阻塞模式转换为交互模式。交互模式下plt.plot(x)或plt.imshow(x)是直接出图象，不需要plt.show()，若在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的化，则图像会一闪而过。
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal') # 绘制网格图，第1，2个参数表示网格个数，第三个参数表示第几个子图

  if not os.path.exists('output'):
    os.makedirs('output')

  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt') # data/train/*/det/det.txt
  for seq_dets_fn in glob.glob(pattern): # 返回所有匹配的文件路径表，参数定义了文件路径匹配规则，可以为绝对路径，也可以为相对路径。(获得该路径下的所有det.txt)

    # create instance of the SORT tracker // 用以计算被跟踪对象的下一帧BBox
    mot_tracker = Sort(max_age=args.max_age, # Amax
                       min_hits=args.min_hits,# 代表前3帧
                       iou_threshold=args.iou_threshold) # IOU阈值

    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',') # 读取 det.txt 文件 // 第0列代表帧数，第2-6列代表物体的BBox。
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0] # split()通过指定分隔符对字符串进行切片。 // os.path.sep表示路径分隔符'/'
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq)) # pattern中分隔出的seq
      for frame in range(int(seq_dets[:,0].max())): # seq_dets第0列最大的值代表了本数据集的总帧数，循环逐帧处理
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7] # 取每一行的第2到第6列，即每一行BBox的坐标
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2] -- (x1,y1)左上角坐标;(x2,y2)右下角坐标
        total_frames += 1

        if(display): # 如果要显示跟踪结果，那么先将数据集中当前帧的jpg文件显示在屏幕上
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time() # 当前时间
        trackers = mot_tracker.update(dets) # 当前帧中所有检测物体的BBox送入SORT算法，获得对所有物体的跟踪计算结果BBox
        cycle_time = time.time() - start_time # 获得SORT算法的耗时
        total_time += cycle_time # 总时间积累

        for d in trackers: # 将SORT更新的所有跟踪结果逐个画到当前帧并显示到屏幕上
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file) # d[4]为ID信息
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
