import subprocess
import re
import numpy as np
base_cmd = 'python main.py rrdt {} --no-display start {},{} goal {},{} --max-number-nodes 5000'
min_max = {
    'rad_mul' :[0.01,10],
    'ndtrees': [2,10],
    'restart': [2,100],
    'energy': [1,100],
    'particles': [0.01,5]
}
hyper_cols = ['rad_mul','ndtrees','restart','energy','particles']
x0 = np.log([1.1,4,20,10,0.1])
ft = [True,False,False,False,True]

class PlanFunc:
    def __init__(self): 

        maps = {'maps/test.png':[[[139,73],[196,156]],[[13,113],[248,73]],[[19,9],[249,248]]],
                'maps/room1.png':[[[397,277],[187,274]],[[91,315],[304,100]],[[460,76],[87,305]]],
                'maps/maze2.png':[[[121,248],[178,167]],[[233,144],[91,223]],[[272,277],[51,40]]]
            }
            # so much faster
        maps = {'maps/test.png':[[[139,73],[196,156]],[[13,113],[248,73]]],
        'maps/maze2.png':[[[121,248],[178,167]],[[233,144],[91,223]]]
        }
        cmds = sum([[base_cmd.format(k,_[0][0],_[0][1],_[1][0],_[1][1]) for _ in v] for k,v in maps.items()],[])
        self.dataset = cmds
        self.balanced_split = np.arange(len(cmds))
        
        self.train_data = [self.dataset[i] for i in self.balanced_split]
        self.test_data = [self.dataset[i] for i in range(len(self.dataset)) if i not in self.balanced_split]
        
        self.results_log = {}
        self.min_vec = 9e12*np.ones(len(self.balanced_split))
    def reset(self):
        self.results_log = {}
        self.min_vec = 9e12*np.ones(len(self.balanced_split))
    def eval_cmd(self,cmd,x):
        res_s = []
        for c,v,f in zip(hyper_cols,x,ft):
            v2 = np.clip(np.exp(v),min_max[c][0],min_max[c][1])
            if not f:
                v3 = int(round(v2))
            else:
                v3 = float(round(v2,2))
            res_s.append('--'+c)
            res_s.append(str(v3))
        result = subprocess.run(cmd.split()+res_s,stdout=subprocess.PIPE)
        try:
            cost = int(re.findall('cc_vi: (\d+)\n',result.stdout.decode())[0])
        except KeyboardInterrupt:
            raise
        except:
            cost = 5000
        return cost

    def joint_func(self,x):
        f_res = np.array([self.eval_cmd(cmd,x) for cmd in self.train_data])
        self.min_vec = np.minimum(self.min_vec,f_res)
        self.results_log[tuple(x)] = f_res
        return f_res.mean()
    def idx_func(self,x,idx=[]):
        f_res = np.array([self.eval_cmd(self.train_data[i],x) for i,_ in enumerate(idx) if _])
        self.min_vec[idx] =  np.minimum(self.min_vec[idx],np.array(f_res))
        return f_res