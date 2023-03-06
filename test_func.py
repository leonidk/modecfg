import numpy as np
from pymoo.problems import get_problem

class ComplicatedFunc:
    def __init__(self,N_SHIFT=5,DIM=2,STD_SAMPLE=10000,seed=42,SCALE_F=1):
        np.random.seed(seed)
        
        p_names = ["ackley","griewank","rastrigin","zakharov"]
        FUNCS = [get_problem(_,n_var=DIM) for _ in p_names]

        F2 = []
        for i,_ in enumerate(FUNCS):
            if _.pareto_set() is not None and len(_.pareto_set()) > 0 and abs(_.pareto_set()[0]).sum() == 0:
                F2.append(_)
            else:
                print(p_names[i],_.pareto_set())
        FUNCS = F2
        N_FUNC = len(FUNCS)
        FUNC_VARS = []
        for func in FUNCS:
            func_var = np.max(func.evaluate(np.random.randn(STD_SAMPLE,DIM)*SCALE_F))
            FUNC_VARS.append(func_var)

        F_TEST = []
        SHIFTS = []
        for i in range(N_SHIFT):
            rand_shift = np.random.randn(DIM)
            rand_rot = np.linalg.svd(np.random.randn(DIM,DIM))[0] * SCALE_F
            SHIFTS.append(rand_shift)
            for func,func_var in zip(FUNCS,FUNC_VARS):
                F_TEST.append((func,func.ideal_point()[0],func_var,rand_shift,rand_rot))
        
        self.F_TEST = F_TEST
        self.DIM = DIM
        self.N_SHIFT = N_SHIFT
        self.N_FUNC = N_FUNC
        self.SHIFTS = SHIFTS
        self.results_log = {}
        self.min_vec = 9e12*np.ones(len(F_TEST))
    def reset(self):
        self.results_log = {}
        self.min_vec = 9e12*np.ones(len(self.F_TEST))
    def joint_func(self,x):
        # np.exp(-self.SHIFT_SCALE*((x-rand_shift)**2).sum() )*
        f_res = [(func.evaluate((x-rand_shift)@rand_rot)-f_best)/func_var for func,f_best,func_var,rand_shift,rand_rot in self.F_TEST]
        f_res = np.array(f_res).reshape((self.N_SHIFT,self.N_FUNC))
        self.min_vec = np.minimum(self.min_vec,f_res.ravel())
        self.results_log[tuple(x)] = f_res
        return f_res.ravel().mean()
    #def sep_func(self,x,idx=0):
    #    return sum([(func.evaluate((x-rand_shift)@rand_rot)-f_best)/func_var for func,f_best,func_var,rand_shift,rand_rot in self.F_TEST[idx*self.N_FUNC:(idx+1)*self.N_FUNC]])
    #def ind_func(self,x,idx=0):
    #    func,f_best,func_var,rand_shift,rand_rot = self.F_TEST[idx]
    #    return (func.evaluate((x-rand_shift)@rand_rot)-f_best)/func_var
    def idx_func(self,x,idx=[]):
        f_test = [self.F_TEST[i] for i,_ in enumerate(idx) if _]
        f_res = [(func.evaluate((x-rand_shift)@rand_rot)-f_best)/func_var for func,f_best,func_var,rand_shift,rand_rot in f_test]
        f_res = np.array(f_res).ravel()
        self.min_vec[idx] =  np.minimum(self.min_vec[idx],np.array(f_res))
        return f_res#.mean()