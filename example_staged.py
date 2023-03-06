import pickle
import time
import os
import argparse

import numpy as np

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
    parser.add_argument("-part", type=str,choices=['partition','cluster'],help="partition method",default='partition')

    parse_res = parser.parse_args()

    cma_popsize = parse_res.popsize
    cma_sigma = parse_res.sigma
    cma_maxf = parse_res.maxf

    N_ITER = parse_res.iter
    out_dir = '{}_results'.format(parse_res.func) if '[func]' in parse_res.outdir else parse_res.outdir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    optimizer_name = parse_res.opt
    ident = 'staged_{}_{:.1f}_{:.1f}_{:d}'.format(optimizer_name,cma_popsize,cma_sigma,cma_maxf)

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
        es = cma.CMAEvolutionStrategy(x0,cma_sigma,{'popsize_factor':cma_popsize,
                                                    'verbose':-9,
                                                    'verb_log':0,
                                                    'maxfevals':cma_maxf//2})
        opt_res = es.optimize(comp_func.joint_func)
        fopt = opt_res.best.f
        xopt = opt_res.best.x


        X = []
        cfgs = []

        for x, f_res in comp_func.results_log.items():
            x_arr = np.array(x)
            X.append(f_res.ravel())
            cfgs.append(x_arr)

        X = np.array(X).T

        import partition_method
        if parse_res.part == 'partition':
            if parse_res.clustn == 2:
                best_cfg_i, c_labels = partition_method.exhaust_partition(np.array(X),2)
            else:
                best_cfg_i, c_labels = partition_method.optimize_partition(np.array(X),parse_res.clustn)
        elif parse_res.part == 'cluster':
                best_cfg_i, c_labels = partition_method.cluster_partition(np.array(X),parse_res.clustn)

        best_cfgs = np.array(best_cfg_i)[c_labels]

        best_f = X[np.arange(X.shape[0]),best_cfgs].mean()

        remain_budget = cma_maxf//2
        total_f = 0
        best_xs = []
        for i in range(2):
            idx = c_labels == i
            local_func = lambda x: comp_func.idx_func(x,idx).mean()
            es2 = cma.CMAEvolutionStrategy(cfgs[best_cfg_i[i]],cma_sigma,{'popsize_factor':cma_popsize,
                                                        'verbose':-9,
                                                        'verb_log':0,
                                                        'maxfevals':cma_maxf//2})
            opt_res2 = es2.optimize(local_func)
            total_f += comp_func.idx_func(opt_res2.best.x,idx).sum()
            best_xs.append(es2.best.x)

        results.append([total_f/len(c_labels),init_res,comp_func.min_vec.mean()])
        method_sols.append((c_labels,best_xs))

    out_time = int(round(time.time()))

    export_data['results'] = results
    export_data['clusters'] = method_sols

    with open('{}/{}_{}.pkl'.format(out_dir,ident,out_time),'wb') as fp:
        pickle.dump(export_data,fp)
