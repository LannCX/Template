'''
    The image and video related utilities.
'''

import os
import cv2
import time
import math
import torch
import random
import numpy as np


def sec_to_frame(sec, fps, st_fr=0):
    return (sec-st_fr)*fps


# read selected frames from the video file, only frame within frame_set are selected...
def fast_read_sel_frames_from_vid_file(fn, frame_set=None, verbose=False):
    all_frames_dict = dict()
    vid_cap=cv2.VideoCapture(fn)
    if frame_set is None:
        vid_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_set = list(range(vid_len))
    cnt=0
    while True:
        ret = vid_cap.grab()
        cnt += 1
        if cnt%500==0 and verbose:
            print(fn, ":", cnt)
        if not ret:
            break

        if cnt in frame_set:
            ret, fr=vid_cap.retrieve()
            if not ret:
                break

            all_frames_dict[cnt] = fr
            frame_set.remove(cnt)

        if 0 in frame_set and cnt==1:
            ret, fr = vid_cap.retrieve()

            if ret==False:
                break

            all_frames_dict[0] = fr
            frame_set.remove(0)
        if len(frame_set) == 0:
            break
    vid_cap.release()
    return all_frames_dict


# image resize
def cv_resize_by_long_edge(im, dst_size=None):
    org_h, org_w, _ = im.shape

    if dst_size is not None:
        if org_h > org_w:
            im = cv2.resize(im, (int(dst_size / float(org_h) * org_w), int(dst_size)))
        else:
            im = cv2.resize(im, (dst_size, int(dst_size / float(org_w) * org_h)))

    return im


def ToImg(raw_flow, bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow += bound
    flow *= (255/float(2*bound))
    return flow


def calc_opticalflow(prev_image, image, bound=20, verbose=False):
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dtvl1 = cv2.createOptFlow_DualTVL1()
    tic = time.time()
    flowDTVL1 = dtvl1.calc(prev_gray, gray, None)
    if verbose:
        print('Optical flow cost time: {}s'.format(time.time()-tic))
    flow = ToImg(flowDTVL1, bound=bound)
    return flow


def uniform_temporal_sampling(sample_duration, all_frames, rand_sel=False, is_sort=True):
    # all_frames = list(range(n_frames))
    n_frames = len(all_frames)
    frame_indices = []
    num_to_sampling = sample_duration
    if sample_duration>n_frames:
        frame_indices = all_frames
        # padding the end frame of video
        while len(frame_indices) < sample_duration:
            frame_indices.append(all_frames[-1])
    else:
        while num_to_sampling > 0:
            remain_frames = [fm for fm in all_frames if fm not in frame_indices]
            sample_stride = int(math.ceil((len(remain_frames)/num_to_sampling)))
            temp_index = list(range(0, len(remain_frames), sample_stride))
            temp_frames = [remain_frames[fm] for fm in temp_index]
            num_to_sampling = num_to_sampling - len(temp_frames)
            frame_indices.extend(temp_frames)
    if rand_sel and random.random() < 0.5 and 2*sample_duration < n_frames:
        comb_set = set(all_frames).difference(set(frame_indices))
        frame_indices = uniform_temporal_sampling(sample_duration, comb_set, is_sort=False, rand_sel=True)

    if is_sort:
        frame_indices.sort()
    assert len(frame_indices)==sample_duration
    return frame_indices
