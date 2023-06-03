import os

import sys
import cv2
import numpy as np
import torch
from PIL import Image
from models.sample_model import SampleFromPoseModel
from utils.language_utils import (generate_shape_attributes,
                                  generate_texture_attributes)
from utils.options import dict_to_nonedict, parse

color_list = [(0, 0, 0), (255, 250, 250), (220, 220, 220), (250, 235, 215),
              (255, 250, 205), (211, 211, 211), (70, 130, 180),
              (127, 255, 212), (0, 100, 0), (50, 205, 50), (255, 255, 0),
              (245, 222, 179), (255, 140, 0), (255, 0, 0), (16, 78, 139),
              (144, 238, 144), (50, 205, 174), (50, 155, 250), (160, 140, 88),
              (213, 140, 88), (90, 140, 90), (185, 210, 205), (130, 165, 180),
              (225, 141, 151)]

def process_image(input_file, shape_text_txt, texture_text_txt, output_file):
    opt = './configs/sample_from_pose.yml'
    opt = parse(opt, is_train=False)
    opt = dict_to_nonedict(opt)
    sample_model = SampleFromPoseModel(opt)

    # Load pose image
    pose_img = Image.open(input_file)
    pose_img = np.array(
        pose_img.resize(
            size=(256, 512),
            resample=Image.LANCZOS))[:, :, 2:].transpose(
                2, 0, 1).astype(np.float32)
    pose_img = pose_img / 12. - 1
    pose_img = torch.from_numpy(pose_img).unsqueeze(1)
    sample_model.feed_pose_data(pose_img)

    # Generate parsing
    f = open(shape_text_txt)
    shape_text = f.read()
    shape_attributes = generate_shape_attributes(shape_text)
    shape_attributes = torch.LongTensor(shape_attributes).unsqueeze(0)
    sample_model.feed_shape_attributes(shape_attributes)
    sample_model.generate_parsing_map()
    sample_model.generate_quantized_segm()
    colored_segm = sample_model.palette_result(
            sample_model.segm[0].cpu())
    # mask_m = cv2.cvtColor(cv2.cvtColor(colored_segm, cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2RGB)
    cv2.imwrite("parsing3.png", colored_segm)

    # Generate human
    f = open(texture_text_txt)
    texture_text = f.read()
    texture_attributes = generate_texture_attributes(texture_text)
    texture_attributes = torch.LongTensor(texture_attributes)
    sample_model.feed_texture_attributes(texture_attributes)
    sample_model.generate_texture_map()
    result = sample_model.sample_and_refine()
    result = result.permute(0, 2, 3, 1)
    result = result.detach().cpu().numpy()
    result = result * 255
    output_img = np.asarray(result[0, :, :, :], dtype=np.uint8)

    # Save output image
    cv2.imwrite(output_file, output_img[:, :, ::-1])

if __name__ == '__main__':
    input_file=sys.argv[1]
    shape_text = sys.argv[2]
    texture_text = sys.argv[3]
    output_file = sys.argv[4]
    process_image(input_file, shape_text, texture_text, output_file)