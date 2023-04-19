import subprocess
import os
import numpy as np
import multiprocessing
import time

folder = '../kissat'
dataset =['Agile/bench_9411.smt2.cnf.bz2', 'Agile/bench_5872.smt2.cnf.bz2',
       'Agile/bench_473.smt2.cnf.bz2', 'Agile/bench_800.smt2.cnf.bz2',
       'Agile/bench_5871.smt2.cnf.bz2', 'Agile/bench_801.smt2.cnf.bz2',
       'Agile/bench_472.smt2.cnf.bz2', 'Agile/bench_4088.smt2.cnf.bz2',
       'Agile/bench_4265.smt2.cnf.bz2', 'Agile/bench_4266.smt2.cnf.bz2',
       'Agile/bench_8080.smt2.cnf.bz2', 'Agile/bench_13514.smt2.cnf.bz2',
       'Agile/bench_8100.smt2.cnf.bz2', 'Agile/bench_13513.smt2.cnf.bz2',
       'Agile/bench_4089.smt2.cnf.bz2', 'Agile/bench_4096.smt2.cnf.bz2',
       'Agile/bench_193.smt2.cnf.bz2', 'Agile/bench_192.smt2.cnf.bz2',
       'Agile/bench_446.smt2.cnf.bz2', 'Agile/bench_4095.smt2.cnf.bz2']

config_settings = [['backboneeffort', 20.0, 0.0, 100000.0, '"effort in per mille"'],
 ['backbonemaxrounds', 1000.0, 1.0, 2147483647.0, '"maximum backbone rounds"'],
 ['backbonerounds', 100.0, 1.0, 2147483647.0, '"backbone rounds limit"'],
 ['bumpreasonslimit',
  10.0,
  1.0,
  2147483647.0,
  '"relative reason literals limit"'],
 ['bumpreasonsrate', 10.0, 1.0, 2147483647.0, '"decision rate limit"'],
 ['chronolevels', 100.0, 0.0, 2147483647.0, '"maximum jumped over levels"'],
 ['compactlim', 10.0, 0.0, 100.0, '"compact inactive limit (in percent)"'],
 ['decay', 50.0, 1.0, 200.0, '"per mille scores decay"'],
 ['definitioncores', 2.0, 1.0, 100.0, '"how many cores"'],
 ['definitionticks', 1000000.0, 0.0, 2147483647.0, '"kitten ticks limits"'],
 ['defraglim', 75.0, 50.0, 100.0, '"usable defragmentation limit in percent"'],
 ['eliminateclslim',
  100.0,
  1.0,
  2147483647.0,
  '"elimination clause size limit"'],
 ['eliminateeffort', 100.0, 0.0, 2000.0, '"effort in per mille"'],
 ['eliminateinit', 500.0, 0.0, 2147483647.0, '"initial elimination interval"'],
 ['eliminateint', 500.0, 10.0, 2147483647.0, '"base elimination interval"'],
 ['eliminateocclim',
  2000.0,
  0.0,
  2147483647.0,
  '"elimination occurrence limit"'],
 ['eliminaterounds', 2.0, 1.0, 10000.0, '"elimination rounds limit"'],
 ['emafast',
  33.0,
  10.0,
  1000000.0,
  '"fast exponential moving average window"'],
 ['emaslow',
  100000.0,
  100.0,
  1000000.0,
  '"slow exponential moving average window"'],
 ['forwardeffort', 100.0, 0.0, 1000000.0, '"effort in per mille"'],
 ['mineffort',
  10.0,
  0.0,
  2147483647.0,
  '"minimum absolute effort in millions"'],
 ['minimizedepth', 1000.0, 1.0, 1000000.0, '"minimization depth"'],
 ['modeinit', 1000.0, 10.0, 100000000.0, '"initial focused conflicts limit"'],
 ['probeinit', 100.0, 0.0, 2147483647.0, '"initial probing interval"'],
 ['probeint', 100.0, 2.0, 2147483647.0, '"probing interval"'],
 ['reducefraction', 75.0, 10.0, 100.0, '"reduce fraction in percent"'],
 ['reduceinit', 1000.0, 2.0, 100000.0, '"initial reduce interval"'],
 ['reduceint', 1000.0, 2.0, 100000.0, '"base reduce interval"'],
 ['rephaseinit', 1000.0, 10.0, 100000.0, '"initial rephase interval"'],
 ['rephaseint', 1000.0, 10.0, 100000.0, '"base rephase interval"'],
 ['restartint', 1.0, 1.0, 10000.0, '"base restart interval"'],
 ['restartmargin', 10.0, 0.0, 25.0, '"fast/slow margin in percent"'],
 ['shrink', 3.0, 0.0, 3.0, '"learned clauses (1=bin', '2=lrg', '3=rec)"'],
 ['substituteeffort', 10.0, 1.0, 1000.0, '"effort in per mille"'],
 ['substituterounds', 2.0, 1.0, 100.0, '"maximum substitution rounds"'],
 ['subsumeclslim',
  1000.0,
  1.0,
  2147483647.0,
  '"subsumption clause size limit"'],
 ['subsumeocclim',
  1000.0,
  0.0,
  2147483647.0,
  '"subsumption occurrence limit"'],
 ['sweepclauses', 1024.0, 0.0, 2147483647.0, '"environment clauses"'],
 ['sweepdepth', 1.0, 0.0, 2147483647.0, '"environment depth"'],
 ['sweepeffort', 10.0, 0.0, 10000.0, '"effort in per mille"'],
 ['sweepfliprounds', 1.0, 0.0, 2147483647.0, '"flipping rounds"'],
 ['sweepmaxclauses',
  4096.0,
  2.0,
  2147483647.0,
  '"maximum environment clauses"'],
 ['sweepmaxdepth', 2.0, 1.0, 2147483647.0, '"maximum environment depth"'],
 ['sweepmaxvars', 128.0, 2.0, 2147483647.0, '"maximum environment variables"'],
 ['sweepvars', 128.0, 0.0, 2147483647.0, '"environment variables"'],
 ['tier1', 2.0, 1.0, 100.0, '"learned clause tier one glue limit"'],
 ['tier2', 6.0, 1.0, 1000.0, '"learned clause tier two glue limit"'],
 ['vivifyeffort', 100.0, 0.0, 1000.0, '"effort in per mille"'],
 ['vivifyirred', 1.0, 1.0, 100.0, '"relative irredundant effort"'],
 ['vivifytier1', 3.0, 1.0, 100.0, '"relative tier1 effort"'],
 ['vivifytier2', 6.0, 1.0, 100.0, '"relative tier2 effort"'],
 ['walkeffort', 50.0, 0.0, 1000000.0, '"effort in per mille"']]

x0 = np.log([_[1]+1 for _ in config_settings])


def sat_eval(params,data):
    MAX_TIME = 15
    low = np.log([_[2]+1 for _ in config_settings])
    high = np.log([_[3]+1 for _ in config_settings])
    params_n = np.exp(np.clip(params,low,high))
    path = os.path.join(folder,data)
    command = os.path.join(folder,'build','kissat')
    extras = ['--{}={}'.format(option[0],int(amount)) for option, amount in zip(config_settings,params_n)]
    cmd_str = [command,path] + extras +['-q','-n','--time={}'.format(MAX_TIME)]
    t1 = time.time()
    res = subprocess.run(cmd_str,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    t2 = time.time()
    #print(res)
    #print(' '.join(cmd_str))
    suc = 'SAT' in str(res.stdout)
    return t2-t1 if suc else MAX_TIME*3

class SATFunc:
    def __init__(self): 
        # using only clean (3, 4, 5, 6, 7, 10)
        # using all 15  ( 1,  2,  3,  6,  8,  9, 11, 13)
        # [ 0,  1,  2,  4,  7, 12, 13, 14, 16, 17, 18, 21, 23, 27, 28, 29, 33, 36, 37, 40, 41, 44, 45]
        # multiprocessing.cpu_count()//2
        self.pool = multiprocessing.Pool(4) # 

        balanced_split = list(range(len(dataset)))
        config_settings
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
        f_res = np.array(self.pool.starmap(sat_eval,[(x,ex_data) for ex_data in self.train_data]))
        self.min_vec = np.minimum(self.min_vec,f_res)
        self.results_log[tuple(x)] = f_res
        return f_res.mean()
    def idx_func(self,x,idx=[]):
        f_res = np.array([sat_eval(x,self.train_data[i]) for i,_ in enumerate(idx) if _])
        self.min_vec[idx] =  np.minimum(self.min_vec[idx],np.array(f_res))
        return f_res