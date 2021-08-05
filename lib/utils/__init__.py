'''
    Common used functions
'''
import os
import time
import json
import errno
import shutil
import pickle
import numpy as np
from time import time

IMG_EXTENSIONS = [
    'jpg', 'JPG', 'jpeg', 'JPEG',
    'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP']

VID_EXTENSIONS = [
    'mp4', 'avi', 'webm']

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def mkdir(path, is_cover=False):
    if not os.path.exists(path) or is_cover:
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def rmdir(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

#   copy all files in src to dst
def copy_all_files_from_src_to_dst(src,dst):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dst)
    return

#   write dict or other objects to file
def write_dict_to_pkl(obj,fn):
    if not fn.endswith(".pkl"):
        fn = fn+".pkl"
    with open(fn, "wb") as f:
        pickle.dump(obj, f)

#   read object from dict
def read_dict_from_pkl(fn):
    if not fn.endswith(".pkl"):
        fn = fn+".pkl"
    try:
        with open(fn, "rb") as f:
            obj = pickle.load(f)
    except UnicodeDecodeError:  # pkl file is writen by python2
        with open(fn, 'rb') as f:
            obj = pickle.load(f, encoding='iso-8859-1')
    return obj

#   write dict or other objects to json
def write_dict_to_json(obj, fn):
    if not fn.endswith(".json"):
        fn = fn+".json"

    json_str = json.dumps(obj, sort_keys=True, indent=2)
    with open(fn, "w") as f:
        f.writelines(json_str)
    return

#   read object from json file
def read_dict_from_json(fn):
    if not fn.endswith(".json"):
        fn=fn+".json"
    with open(fn) as json_file:
        obj = json.load(json_file)

    return obj

#   write dict or other objects to both pkl and json format
def write_dict_to_pkl_and_json(obj,fn):
    fn=fn.replace(".json","")
    fn=fn.replace(".pkl","")

    json_fn=fn+".json"
    pkl_fn=fn+".pkl"

    write_dict_to_json(obj,json_fn)
    write_dict_to_pkl(obj,pkl_fn)
    return

def getFileCreateInfo(filepath):
    create_time = time.localtime(os.stat(filepath).st_ctime)
    #ctime = time.strftime('%Y-%m-%d %H:%M:%S',create_time)
    return [create_time.tm_hour, create_time.tm_min]

#   running mean calculation with x and window size N
def runningMeanFast(x, N):
    org_x_len=len(x)
    to_append=x[-N::]
    x=np.concatenate((x,to_append),axis=0)
    conv_x=np.convolve(x, np.ones((N,))/N)[(N-1):]
    ret_x=conv_x[0:org_x_len]
    return ret_x

# from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

