# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import os
import cv2

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

fr = 5
im_fn1 = os.path.expanduser('~/datasets/egopose/mocap1205/fpv_frames/take_01/%05d.png' % (fr + 1))
im_fn2 = os.path.expanduser('~/datasets/egopose/mocap1205/fpv_frames/take_01/%05d.png' % fr)
im1 = np.array(Image.open(im_fn1))
im2 = np.array(Image.open(im_fn2))
# im1 = np.array(Image.open('examples/car1.jpg'))
# im2 = np.array(Image.open('examples/car2.jpg'))
im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
e = time.time()
print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
flow_gt = np.load(os.path.expanduser('~/datasets/egopose/mocap1205/fpv_of/take_01/%05d.npy' % fr))
diff_flow = flow - flow_gt
print(np.min(diff_flow), np.max(diff_flow))
np.save('examples/outFlow.npy', flow)

hsv = np.zeros(im1.shape, dtype=np.uint8)
hsv[:, :, 0] = 255
hsv[:, :, 1] = 255
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('examples/outFlow_new.png', rgb)
cv2.imwrite('examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)

flow_vis_gt = cv2.imread(os.path.expanduser('~/datasets/egopose/mocap1205/fpv_of/take_01/%05d.png' % fr))
cv2.imshow('pyflow', rgb)
cv2.imshow('pwc_net', flow_vis_gt)
cv2.moveWindow('pyflow', 0, 60)
cv2.moveWindow('pwc_net', 224, 60)
cv2.waitKey()
