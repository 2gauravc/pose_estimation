import argparse
import logging
import time

import cv2
import numpy as np
import ffmpy
import pickle
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from conn_test import conv_humans_to_recs
from conn_test import insert_pose_keypoints_humans

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def convert_file(inputted_file):
    video_name = "in_file.avi"
    ff = ffmpy.FFmpeg(inputs={inputted_file : None}, outputs={video_name: ' -c:a mp3 -c:v mpeg4'})
    ff.cmd
    ff.run()
    return video_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    #file_to_read = convert_file(args.camera)
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    print("Num Frames:", int(cv2.VideoCapture.get(cam, property_id)))
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    frame_num=0
    while True:
        ret_val, image  = cam.read()
        if ret_val==True: 
            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            logger.debug('postprocess+')
            
            insert_pose_keypoints_humans(humans,1,1)
            logger.debug('db_write+')            
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            logger.debug('show+')
            cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
            #cv2.imshow('tf-pose-estimation result', image)
            out.write(image)
            fps_time = time.time()
            print("processed frame: ", frame_num)
            frame_num+=1
            if frame_num > 50: #cv2.waitKey(1) == 27:
                print("Exiting after 50 frames")
                break
        else:
                break
    
    logger.debug('finished+')
    out.release()
    cv2.destroyAllWindows()
