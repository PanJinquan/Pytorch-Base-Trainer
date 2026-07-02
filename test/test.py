# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 
# @Date   : 2026-07-02 08:58:17
# @Brief  :
# --------------------------------------------------------
"""
from pybaseutils import file_utils,image_utils

if __name__ == '__main__':
    image_file = "data/test.jpg"
    image = image_utils.read_image(image_file)
    print(image.shape)
