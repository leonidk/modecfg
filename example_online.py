import pickle
import time
import os
import argparse

import numpy as np

from bandit import bandit
from copy import deepcopy

import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-func", type=str,choices=['stereo','synth','slam','planner','fuzzy'],help="choose a problem",default='synth')

    parser.add_argument("-opt", type=str,choices=['bo','cma'],help="choose a method",default='cma')
    parser.add_argument("-popsize", type=float, help="CMA-ES popsize multiplier",default=1.0)
    parser.add_argument("-sigma", type=float,help="CMA-ES sigma",default=0.5)
    parser.add_argument("-maxf", type=int, help="maximum function evaluations",default=150)
    parser.add_argument("-iter", type=int, help="number of times to run",default=45)
    parser.add_argument("-outdir", type=str, help="output folder",default='[func]_results')

    parser.add_argument("-scalef", type=float,help="scale for function",default=1.0)
    parser.add_argument("-noph",help="do not run posthoc",action="store_false")
    parser.add_argument("-clustn", type=int, help="clusters",default=2)
    parser.add_argument("-dim", type=int, help="dim for synth func",default=10)
    parser.add_argument("-gen_window", type=int, help="number of generations to use for bandit window",default=2)
    parser.add_argument("-part", type=str,choices=['partition','cluster'],help="partition method",default='partition')

    parse_res = parser.parse_args()

    cma_popsize = parse_res.popsize
    cma_sigma = parse_res.sigma
    cma_maxf = parse_res.maxf

    N_ITER = parse_res.iter
    out_dir = '{}_results'.format(parse_res.func) if '[func]' in parse_res.outdir else parse_res.outdir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    optimizer_name = 'cma'
    ident = 'onlinewin_{}_{:.1f}_{:.1f}_{:d}'.format(optimizer_name,cma_popsize,cma_sigma,cma_maxf)

    export_data = {}
    results = []
    method_sols = []

    if parse_res.func == 'synth':
        from test_func import ComplicatedFunc
        comp_func = ComplicatedFunc()
        x0 = np.zeros(parse_res.dim)
    elif parse_res.func == 'stereo':
        from stereo_func import StereoFunc, x0
        comp_func = StereoFunc()
    elif parse_res.func == 'planner':
        from planner_func import PlanFunc, x0
        comp_func = PlanFunc()
    elif parse_res.func == 'fuzzy':
        from fuzzy_func import FuzzyFunc, x0
        comp_func = FuzzyFunc()
    elif parse_res.func == 'slam':
        from slam_func2 import SLAMFunc2, x0
        comp_func = SLAMFunc2()

    init_res = comp_func.joint_func(x0)
    for i in tqdm.tqdm(range(N_ITER), desc=" inner", position=1, leave=False):
        np.random.seed(42+173+i)
        comp_func.reset()
        import cma
        N_solvers = parse_res.clustn
        es_set = [cma.CMAEvolutionStrategy(x0,cma_sigma,{'popsize_factor':cma_popsize,
                                        'verbose':-9,
                                        'verb_log':0,}) for _ in range(N_solvers)]
        f_evals = 0
        first_run = True
        bandits = []
        
        while f_evals < max(1,cma_maxf-N_solvers*2):
            if first_run:
                solutions = es_set[0].ask()
                res = [comp_func.joint_func(x) for x in solutions]
                es_set[0].tell(solutions, res)
                for i in range(1,N_solvers):
                    es_set[i] = deepcopy(es_set[0])
                first_run = False
                f_evals += len(solutions)
                
                X = np.array(list(comp_func.results_log.values())).reshape((-1,len(comp_func.train_data)))
                for vec in X.T:
                    bandits.append(bandit(N_solvers,list(vec)))
            else:
                idx = np.array([_.decision() for _ in bandits])
                
                sol_n = 0
                for i, es in enumerate(es_set):
                    solutions = es.ask()
                    sol_n = max(sol_n,len(solutions))
                    bin_idx = idx==i
                    res = [comp_func.idx_func(x,bin_idx) for x in solutions]
                    for b_idx, v in zip(np.where(bin_idx)[0],np.array(res).T):
                        for i_v in v:
                            bandits[b_idx].getReward(i_v)

                    es.tell(solutions, [_.mean() for _ in res])
                f_evals += sol_n
        #cfgs_to_full_try = [_.best.x for _ in es_set] + [_.mean for _ in es_set]
        #[comp_func.joint_func(x) for x in cfgs_to_full_try]
        window_size = sol_n*parse_res.gen_window
        idx_choices = np.array([np.argmin(_.getMeans(window_size)) for _ in bandits])

        total_val = 0
        res_sets = []
        for i in range(N_solvers):
            b_idx = (i == idx_choices)
            import numpy as np
            poss_opt = [(comp_func.idx_func(x,b_idx).sum(),np.random.randn(),x) for x in [es_set[i].mean,es_set[i].best.x]]
            try:
                total_val += min(poss_opt)[0]
            except:
                print(poss_opt)
                raise
            res_sets.append(min(poss_opt)[2])
        total_f = total_val / len(bandits)

        results.append([total_f,init_res,comp_func.min_vec.mean()])
        method_sols.append((idx_choices,res_sets))
    out_time = int(round(time.time()))
    export_data['results'] = results
    export_data['clusters'] = method_sols

    with open('{}/{}_{}.pkl'.format(out_dir,ident,out_time),'wb') as fp:
        pickle.dump(export_data,fp)
