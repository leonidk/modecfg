import os
import numpy as np
import subprocess
import re
import itertools
import tempfile
import multiprocessing
from copy import deepcopy

#base_cmd = '/home/lkeselma/code/dso/build/bin/dso_dataset'
base_cmd = '/home/lkeselma/code/LDSO/bin/run_dso_tum_mono'

base_dir = '/home/lkeselma/code/tumvi/'

base_setting_s = {'files': '/home/lkeselma/code/tumvi/dataset-corridor1_512_16/dso/cam0/images',
 'calib': '/home/lkeselma/code/tumvi/dataset-corridor1_512_16/dso/cam0/camera.txt',
 'gamma': '/home/lkeselma/code/tumvi/dataset-corridor1_512_16/dso/cam0/pcalib.txt',
 'vignette': '/home/lkeselma/code/tumvi/dataset-corridor1_512_16/dso/cam0/vignette.png',
 'vocab': '/home/lkeselma/code/LDSO/vocab/orbvoc.dbow3',
 'mode': '0',
 'nogui': '1',
 'quiet': '1',
 'immature': '500',
 'point': '1000',
 'minframe': '5',
 'maxframe': '7',
 'minopt': '1',
 'maxopt': '6',
 'width': '192',
 'height': '192',
 'nolog' : '1',
 'loopclosing' : '0',
 'nomt': '1'}

min_max = {
    'immature': (100,5000),
    'point': (100,5000),
    'minframe': (1,10),
    'maxframe': (5,30),
    'maxopt': (1,16)
}

opt_cols = ['immature','point','minframe','maxframe','maxopt']
x0 = np.log([500,1000,3,5,3])
def eval_folder(dataset_name,x):
    with tempfile.TemporaryDirectory() as path:
        os.chdir(path)
        dataset_path = os.path.join(base_dir,dataset_name)
        os.environ["GLOG_minloglevel"] = "3"
        try:
            base_setting = deepcopy(base_setting_s)
            base_setting['files'] = os.path.join(dataset_path,'dso','cam0','images')
            base_setting['calib'] = os.path.join(dataset_path,'dso','cam0','camera.txt')
            base_setting['gamma'] = os.path.join(dataset_path,'dso','cam0','pcalib.txt')
            base_setting['vignette'] = os.path.join(dataset_path,'dso','cam0','vignette.png')

            for k,v in zip(opt_cols,x):
                if k=='size':
                    im_size = np.exp(v)
                    im_size= round(im_size/8)*8
                    base_setting['height'] = int(round(np.clip(im_size,min_max[k][0],min_max[k][1])))
                    base_setting['width'] = int(round(np.clip(im_size,min_max[k][0],min_max[k][1])))
                else:
                    if k not in base_setting:
                        raise
                    base_setting[k] = int(round(np.clip(np.exp(v),min_max[k][0],min_max[k][1])))

            if os.path.exists('result.txt'):
                subprocess.run(['rm','result.txt'])
            base_setting['maxframe'] = max(base_setting['maxframe'] ,base_setting['minframe'] )
            
            # first bit disables address random, which reduces crashes for DSO
            iter_command = ['setarch','x86_64','-R'] + [base_cmd] + ['{}={}'.format(k,v) for k,v in base_setting.items()]
            #print(' '.join(iter_command))
            run_res = subprocess.run(iter_command, capture_output=True)
            std_out_res = run_res.stdout.decode()
            runtime = [float(_[0]) for _ in re.findall("\n([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))ms per frame \(single core",std_out_res)][0]
            os.system(' '.join(['tr','-s',"'[:space:]'",'<','result.txt','>','result_clean.txt']))
            err_res = subprocess.run(['evo_ape','euroc',os.path.join(dataset_path,'dso','gt_imu.csv'), 'result_clean.txt','-s','-a'],stdout=subprocess.PIPE)
            subprocess.run(['rm','result_clean.txt'])
            eval_error = float(re.findall('mean\t([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))\n',err_res.stdout.decode())[0][0])

            wc_ref = subprocess.run(['wc','-l',os.path.join(dataset_path,'dso','cam0','times.txt')],capture_output=True)
            #wc_m = subprocess.run(['wc','-l','result_clean.txt'],capture_output=True)
            #wc_m = int(wc_m.stdout.decode().split(' ')[0])

            wc_ref = int(wc_ref.stdout.decode().split(' ')[0])
            wc_m = [int(_) for _ in re.findall("\n(\d+) Frames \(",std_out_res)][0]
            # print(wc_ref,wc_m,eval_error)
            if wc_m+2 < wc_ref: # slam cut short. 1 frame for ref. 1 row for headers in times.txt
                eval_error = wc_ref/wc_m # times 1m presumed

        except KeyboardInterrupt:
            raise
        except Exception as e:
            #print(' '.join([base_cmd] + ['{}={}'.format(k,v) for k,v in base_setting.items()]))
            #raise
            print(e)
            print(' '.join(iter_command))
            print(run_res.stdout.decode())
            print(run_res.returncode)
            if run_res.stderr is not None:
                print(run_res.stderr.decode())
            return 9999
        return eval_error*runtime

class SLAMFunc:
    def __init__(self):
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count()-2)
        os.environ["GLOG_minloglevel"] = "3"
        dataset = sorted([_ for _ in os.listdir(base_dir) if 'dataset-' in _ and os.path.isdir(os.path.join(base_dir,_))])
        dataset = [_ for _ in dataset if ('1_512_' in _ or '2_512_' in _ or '3_512_' in _) ]
        balanced_split = [_ for _ in range(len(dataset)) if (_%3)==2]
        #balanced_split = [0, 2, 3, 5, 6, 8, 10, 11]
        #balanced_split = [0, 3, 6, 11]
        #print('\n'.join(dataset))
        #print(balanced_split)
        #raise
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
        f_res = np.array(self.pool.starmap(eval_folder,[(ex_data,x) for ex_data in self.train_data]))
        self.min_vec = np.minimum(self.min_vec,f_res)
        self.results_log[tuple(x)] = f_res
        return f_res.mean()
    def idx_func(self,x,idx=[]):
        f_res = np.array([eval_folder(self.train_data[i],x) for i,_ in enumerate(idx) if _])
        self.min_vec[idx] =  np.minimum(self.min_vec[idx],np.array(f_res))
        return f_res
