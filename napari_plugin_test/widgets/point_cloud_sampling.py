#!/usr/bin/env python3
# coding: utf-8
from magicgui import magicgui
import numpy as np
from skimage import feature
from napari.types import LabelsData, ImageData, PointsData
from napari.layers import Points, Layer
from enum import Enum
import napari
from magicgui.tqdm import trange

class Color(Enum):
    red = 'red'
    green = 'green'
    blue = 'blue'
    yellow = 'yellow'
    black = 'black'
    white = 'white'

@magicgui(call_button='Sample')
def point_cloud_sampling(input: LabelsData,
                         sampling_frequency: float=0.01,
                         sigma: float=1.0,
                         edge_color=Color.black,
                         face_color=Color.red,
                         point_size: int=5) -> Points:
    point_lst = []
    for z in trange(input.shape[0]):
        img = (input[z] > 0).astype('uint8') * 255
        img = feature.canny(img, sigma=sigma)
        points = np.where(img == 1)
        # Just keep values at every n-th position
        for i in range(len(points[0])):
            if np.random.rand() < sampling_frequency:
                point_lst.append([z, points[0][i], points[1][i]])

    points_layer = Points(np.asarray(point_lst))
    points_layer.edge_color = edge_color.value
    points_layer.face_color = face_color.value
    points_layer.size = point_size

    return points_layer
