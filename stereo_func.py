import re, os

import numpy as np
import cv2
from PIL import Image
import pickle
import multiprocessing
def load_pfm(fname):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    file = open(fname,'r',encoding='iso-8859-1')
    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(np.reshape(data, shape)), scale

def load_calibtxt(fname):
    file = open(fname,'rt',encoding='iso-8859-1')
    return {l.split('=')[0]:' '.join(l.split('=')[1:]).rstrip() for l in [line for line in file]}


def load_dataset(path='data/middlebury/mid_out2'):
    dataset = []
    for folder in sorted(os.listdir(path)):
        if '.pkl' not in folder:
            continue
        try:
            with open(os.path.join(path,folder), "rb") as f: 
                out_d = pickle.load(f)
            max_disp = out_d['max_disp']
            imL = np.pad(out_d['im0'], [(0,0), (max_disp,0), (0,0)],mode='constant')
            imR = np.pad(out_d['im1'], [(0,0), (max_disp,0), (0,0)],mode='constant')
            gt = out_d['gt']
            mask_invalid = out_d['mask']
            dataset.append( (imL,imR,max_disp,mask_invalid,gt) )
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            continue
    return dataset
def stereo_match(params,data,max_error=3):
    try:
        imL,imR,max_disp,mask_invalid,gt = data
        p = {
            'blockSize': 5,
            'P1':8 * 3 * 3 ** 2,
            'P2':32 * 3 * 3 ** 2,
            'disp12MaxDiff': 40,
            'uniquenessRatio': 15,
            'speckleWindowSize': 0,
            'speckleRange': 2,
            'preFilterCap': 63,
        }

        p['minDisparity'] = 0
        p['numDisparities'] = max_disp
        p['mode'] = cv2.STEREO_SGBM_MODE_SGBM
        cols = ['P1', 'P2', 'blockSize', 'disp12MaxDiff', 'preFilterCap', 'speckleRange', 'speckleWindowSize', 'uniquenessRatio']

        for c,v in zip(cols,params):
            p[c] = int(round(np.exp(v)))

        left_matcher = cv2.StereoSGBM_create(**p)

        displ = left_matcher.compute(imL, imR)[:,max_disp:]
        result = (displ).astype(np.float32)/16


        err_vec = np.nan_to_num(abs(result-gt))
        err_vec[err_vec > max_error] = max_error
        err_vec[displ == -1] = max_error

        err = np.mean(err_vec[~mask_invalid])
        return err
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        return max_error
    
x0 = np.log(np.array([216, 864, 5, 40, 63, 2, 0.1, 15]))

class StereoFunc:
    def __init__(self,path='data/middlebury/mid_out2',
                    balanced_split = (0,  1,  2,  4,  7, 12, 13, 14, 16, 17, 18, 21, 23, 27, 28, 29, 33, 36, 37, 40, 41, 44, 45) ): 
        # using only clean (3, 4, 5, 6, 7, 10)
        # using all 15  ( 1,  2,  3,  6,  8,  9, 11, 13)
        # [ 0,  1,  2,  4,  7, 12, 13, 14, 16, 17, 18, 21, 23, 27, 28, 29, 33, 36, 37, 40, 41, 44, 45]
        dataset = load_dataset(path)
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count()//2) # 

        #balanced_split = list(range(len(dataset)))

        self.train_data = [dataset[i] for i in balanced_split]
        self.test_data = [dataset[i] for i in range(len(dataset)) if i not in balanced_split]

        self.dataset = dataset
        self.balanced_split = balanced_split

        self.results_log = {}
        self.min_vec = 9e12*np.ones(len(self.balanced_split))
    def reset(self):
        self.results_log = {}
        self.min_vec = 9e12*np.ones(len(self.balanced_split))
    def joint_func(self,x):
        f_res = np.array(self.pool.starmap(stereo_match,[(x,ex_data) for ex_data in self.train_data]))
        #f_res = np.array([stereo_match(x,ex_data) for ex_data in self.train_data])
        self.min_vec = np.minimum(self.min_vec,f_res)
        self.results_log[tuple(x)] = f_res
        return f_res.mean()
    def idx_func(self,x,idx=[]):
        f_res = np.array([stereo_match(x,self.train_data[i]) for i,_ in enumerate(idx) if _])
        self.min_vec[idx] =  np.minimum(self.min_vec[idx],np.array(f_res))
        return f_res