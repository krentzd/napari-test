#!/usr/bin/env python
# -*- coding: utf-8 -*-
from napari_plugin_engine import napari_hook_implementation
from .widgets.clean_binary_segmentation import make_clean_binary_segmentation
from .widgets.log_segmentation import make_log_segmentation
from .widgets.data_preprocessing import make_data_preprocessing

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [(make_log_segmentation, {"name": "Segment FM data"}),
            (make_clean_binary_segmentation, {"name": "Clean segmentations"}),
            (make_data_preprocessing, {"name": "Preprocess data"})]
