#!/usr/bin/env python3
# coding: utf-8
from probreg import cpd, bcpd, callbacks
import numpy as np
import open3d as o3
import transforms3d as t3d
import time
from napari.layers import Points
from magicgui import magicgui
import matplotlib.pyplot as plt
from scipy import ndimage
import napari
import copy

class RegistrationProgressCallback(object):
    def __init__(self, maxiter):
        self.counter = 0
        self.maxiter = maxiter

    def __call__(self, *args):
        self.counter += 1
        print('{}/{}'.format(self.counter, self.maxiter))

def prepare_source_and_target_nonrigid_3d(source_array,
                                          target_array,
                                          voxel_size=5,
                                          every_k_points=2):
    source = o3.geometry.PointCloud()
    target = o3.geometry.PointCloud()
    source.points = o3.utility.Vector3dVector(source_array)
    target.points = o3.utility.Vector3dVector(target_array)
    source = source.voxel_down_sample(voxel_size=voxel_size)
    target = target.voxel_down_sample(voxel_size=voxel_size)
    source = source.uniform_down_sample(every_k_points=every_k_points)
    target = target.uniform_down_sample(every_k_points=every_k_points)
    return source, target

@magicgui(call_button='Register')
def point_cloud_registration(moving: Points,
                             fixed: Points,
                             viewer: 'napari.viewer.Viewer',
                             voxel_size: float=5,
                             every_k_points: int=2,
                             max_iterations: int=50,
                             visualise: bool=False,) -> Points:
    start = time.time()
    source, target = prepare_source_and_target_nonrigid_3d(moving.data,
                                                           fixed.data,
                                                           voxel_size=voxel_size,
                                                           every_k_points=every_k_points)

    viewer.add_points(np.asarray(source.points),
                      name='moving_points',
                      size=5,
                      face_color='red')

    viewer.add_points(np.asarray(target.points),
                      name='fixed_points',
                      size=5,
                      face_color='green')
    cbs = []
    cbs.append(RegistrationProgressCallback(max_iterations))
    if visualise:
        cbs.append(callbacks.Open3dVisualizerCallback(np.asarray(source.points), np.asarray(target.points)))
    tf_param = bcpd.registration_bcpd(source, target, maxiter=max_iterations, callbacks=cbs)
    elapsed = time.time() - start
    print("time: ", elapsed)
    print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rigid_trans.rot)),
          tf_param.rigid_trans.scale, tf_param.rigid_trans.t, tf_param.v)

    transformed = tf_param._transform(source.points)
    transformed_pnts = Points(transformed)
    transformed_pnts.edge_color = moving.edge_color
    transformed_pnts.face_color = 'blue'
    transformed_pnts.size = 5
    transformed_pnts.name = 'transformed_points'
    return transformed_pnts
