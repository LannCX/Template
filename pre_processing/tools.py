import os
import re
import json
import shutil
import pandas as pd
from utils import VID_EXTENSIONS
from decord import VideoReader

gt_file_format = '/raid/chenxu/Code/STImage/pre_processing/kinetics/annotations/kinetics-full_%s.csv'


def count_all(data_root='/raid/chenxu/workspace/Kinetics-400'):
    count=0
    n_fr = []
    for root, dirs, files in os.walk(data_root):
        valid_vid = [x for x in files if x.split('.')[-1] in VID_EXTENSIONS]
        # count+=len(valid_vid)
        for fi in valid_vid:
            vid_path = os.path.join(root, fi)
            try:
                vr = VideoReader(vid_path)
                if len(vr)>10000:
                    print(vid_path)
                n_fr.append(len(vr))
            except:
                continue
            count += 1
    print('total videos: %d, avg: %f, min:%f, max: %f.' % (count, sum(n_fr)/count, min(n_fr), max(n_fr)))


def count(root_path='/raid/chenxu/workspace/Kinetics-400'):
    check_set = ['train', 'val']
    d = {}
    for split in check_set:
        total = 0
        for root, dirs, files in os.walk(os.path.join(root_path,split)):
            total+=len(files)
            for file in files:
                file_path = os.path.join(root, file)
                if not file.endswith('.mp4'):
                    vid_id = file.split('.')[0]
                    dst_path = os.path.join(root,vid_id+'.mp4')
                    shutil.move(file_path, dst_path)
                    print(file)
        d[split] = total
    print(d)


def check_videos(root_path='/raid/chenxu/workspace/Kinetics-400'):
    check_set = ['val', 'train']
    for split in check_set:
        miss = []
        file_name = gt_file_format % split
        df = pd.read_csv(file_name)
        total= 0
        for i in range(df.shape[0]):
            vid_id = df['youtube_id'][i]
            label = df['label'][i].replace(' ','_')
            start = str(int(df['time_start'][i])).zfill(6)
            end = str(int(df['time_end'][i])).zfill(6)
            if split == 'train':
                vid_path = os.path.join(root_path, split, label,
                                        '_'.join([vid_id,start,end])+'.mp4')
            else:
                vid_path = os.path.join(root_path, split, label, vid_id+'.mp4')

            if not os.path.exists(vid_path):
                miss.append(vid_id)
            else:
                total+=1
        print('total %d, miss %d/%d videos in %s'%(total, len(miss), df.shape[0], split))


def gen_class_label(version='400'):
    file_name = 'annotations/kinetics-%s_train.csv' % version

    df = pd.read_csv(file_name)
    classes = set(df['label'].to_list())
    class_id = {c:i for i,c in enumerate(classes)}
    with open('class_id-%s.json'%version, 'w') as f:
        json.dump(class_id,f,indent=2)


def gen_hmdb_label(path='/raid/chenxu/Code/STImage/pre_processing/hmdb51/annotations'):
    names = os.listdir(path)
    file_names = [re.sub('_test_split\d{1}\.txt', '', x) for x in names]
    class_id = {n: i for i, n in enumerate(set(file_names))}
    with open('class_id.json', 'w') as f:
        json.dump(class_id, f, indent=2)


def gen_dtdb_label(path='/data/yangfeng/DTDB/BY_DYNAMIC_FINAL/TEST'):
    names = os.listdir(path)
    class_id = {n: i for i, n in enumerate(set(names))}
    with open('class_id_DYNAMIC.json', 'w') as f:
        json.dump(class_id, f, indent=2)
    # Train: total videos: 6488, avg: 378.886252, min:3.000000, max: 10035.000000.
    # Test: total videos: 2709, avg: 248.665559, min:54.000000, max: 911.000000.


def minikinetics():
    out_format = gt_file_format.replace('400', '200')
    # split the mini kinetics data
    for split in ['train', 'val']:
        org_file_name = gt_file_format % split
        df = pd.read_csv(org_file_name)
        out_file = out_format%split
        txt_file = out_format.replace('csv','txt')%split
        with open(txt_file, 'r') as f:
            data = f.readlines()
            data = [x.replace('\n', '') for x in data]
        out_df = df[df['youtube_id'].isin(data)]
        out_df.to_csv(out_file, index=False)
    return


if __name__ == '__main__':
    # gen_dtdb_label()
    # count()
    # check_videos()
    # gen_class_label()
    # minikinetics()
    # data = pd.read_csv('annotations/kinetics-200_val.csv')
    # print('OK')
    count_all('/data/yangfeng/DTDB/BY_DYNAMIC_FINAL/TRAIN')
    # gen_hmdb_label()
