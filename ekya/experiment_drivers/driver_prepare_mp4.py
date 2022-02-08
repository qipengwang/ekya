import argparse
import glob
import os

import pandas as pd

from ekya.datasets.Mp4VideoClassification import Mp4VideoClassification

DEVICE = 2


# def parse_args():
#     parser = argparse.ArgumentParser("Prepare Bellevue dataset",
#                                      description="Ekya script.")
#
#     parser.add_argument("--video_dir",  type=str,
#                         help="Directory where all mp4 videos sit.")
#     parser.add_argument("--target_fps",  type=float, default=0.5,
#                         help="Target fps the video will be encoded.")
#     parser.add_argument("--save_dir",  type=str, help="Directory where "
#                         "processed mp4 videos will be saved.")
#     args = parser.parse_args()
#
#     return args

dataset_name = 'mp4'
# dataset_name = 'bellevue'
# root = os.path.join('/data/zxxia/ekya/datasets', dataset_name)
root = os.path.join('/data3/zxxia/', dataset_name)

# vid_name = 'Bellevue_Bellevue_NE8th__2017-09-10_18-08-23'
# vid_name = 'Bellevue_150th_SE38th__2017-09-11_13-08-32'
# vid_path = os.path.join(root,  f'{vid_name}.mp4')

vid_paths = glob.glob(os.path.join(root,  '*.mp4'))

for vid_path in sorted(vid_paths):
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    # create frame image folder
    img_dir = os.path.join(root, vid_name, 'frame_images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        continue

    # create classificaiton image folder
    clf_img_dir = os.path.join(root, vid_name, 'classification_images')
    if not os.path.exists(clf_img_dir):
        os.makedirs(clf_img_dir)


    # extract frames images
    Mp4VideoClassification.extract_frame_images(vid_path, img_dir)

    # generate object detection
    Mp4VideoClassification.generate_object_detection(
        vid_path, img_dir, os.path.join(root, vid_name),
        '/data/zxxia/ekya/faster_rcnn_resnet101_coco_2018_01_28', DEVICE)

    root = os.path.join('/data/zxxia/ekya/datasets', dataset_name)
    Mp4VideoClassification.generate_sample_list_from_detection_file(
        vid_path,
        os.path.join(root, vid_name, f'{vid_name}_detections.csv'),
        os.path.join(root, 'sample_lists', 'citywise'), min_res=30)
