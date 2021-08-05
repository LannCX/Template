import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


STHV1_DATA = [
    ['ECO_8F', 47.5, 32, 39.6],
    ['ECO_16F', 47.5, 64, 41.4],
    # ['ECO_En_Lite', 150, 267, 46.4],
    ['I3D', 28, 306, 41.6],
    ['I3D_NL', 35.3, 336, 44.4],
    ['I3D_NL+GCN', 62.2, 606, 46.1],
    ['TSM', 24.3, 33, 45.6],
    # ['TSM_16F', 24.3, 65, 47.2],
    ['TSN', 24.3, 65, 19.7],
    ['MultiScale TRN', 18.3, 16, 34.4],
    # ['TSM_En', 48.6, 98, 49.7],
    # ['TEA_8F', 26.149, 35, 48.9],
    # ['TEA_16F', 26.149, 70, 51.9],
    # ['GSM_8F', 10, 16.46, 47.24],
    # ['GSM_16F', 10, 32.92, 49.56],
    # ['RubiksNet', 8.5, 15.8, 46.4],
    ['Ours', 5.6, 18, 45.5],
]
DTDB_DATA = [
    ['C3D', 27.66, 154.367, 75.5],
    ['S3D', 8.014, 18.067, 68.1],
    ['TSN-BNInception', 10, 32.92, 73.2],
    ['TSN-ResNet', 24.3, 65, 73.49],
    ['TSM', 24.3, 65, 74.6],
    ['Ours', 5.6, 18.31, 82.17],
]
PARAM_TH = [10, 30, 50, 150]
COLORS = [x for x in list(mcolors.TABLEAU_COLORS.keys()) if x!='gray']
CACHE_COLOR = {'Ours': 'red'}


def heatmap_caliberation(cam):
    cam=cam-np.min(cam)
    cam=cam/np.max(cam)
    cam=np.uint8(255*cam)
    return cam


def viz_heatmap_on_img(im, heatmap, hm_weight=0.6):
    pass
    heatmap = heatmap_caliberation(heatmap)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = hm_weight * colormap + (1.0-hm_weight) * im
    return result


def random_color(pre_suffix):
    if pre_suffix not in CACHE_COLOR:
        color = random.choice(COLORS)
        while color in CACHE_COLOR.values():
            color = random.choice(COLORS)
        CACHE_COLOR[pre_suffix] = color
    else:
        color = CACHE_COLOR[pre_suffix]
    return color


def plot_pareto():
    fig = plt.figure()
    plt.rc('font', family='Times New Roman')
    plt.rc('axes', axisbelow=True)  # 网格置于底层
    plt.xlabel('FLOPs/Video (G)')
    plt.ylabel('Accuracy (%)')
    plt.grid(linestyle='-.')
    plt.xlim([0, 700])
    # plt.ylim([35, 55])

    for d in STHV1_DATA:
        size = 10*int(d[1])
        # delta_x = int(0.1*size)
        # delta_y = int(0.014*size)
        color = random_color(d[0].split('_')[0])
        plt.scatter(d[2]-20, d[3]+1, s=size, c=color)
        if d[0] =='Ours':
            plt.annotate(xy=(d[2]-40, d[3]-0.8), text=d[0], size=10, color='red')
        else:
            plt.annotate(xy=(d[2], d[3]), text=d[0], size=10, color='k')
        # plt.annotate(xy=(d[2]-30, d[3]), text=d[0], size=10, color='k')
    # plt.annotate(text='#Parameters', xy=(10, 49.4), size=10, color='k')

    # plot references
    # ax = fig.add_axes([0.125, 0.72, 0.34, 0.16])
    # ax = fig.add_axes([0.56, 0.63, 0.34, 0.2])
    ax = fig.add_axes([0.56, 0.13, 0.34, 0.2])
    ax.set(xlim=[2, 14], xticks=[], yticks=[], title='#Parameters')
    for i, r in enumerate(PARAM_TH):
        size = 10 * int(r)
        ax.scatter((i+1)*3, 50, s=size, color=(0.9, 0.9, 0.9))
        ax.annotate(xy=((i + 1) * 3 - 0.55, 50), text=str(r) + 'M', size=8, color='k')

    plt.show()

