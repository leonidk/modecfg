import numpy as np
import pykitti
import jax
import jax.numpy as jnp

import fm_render
import sklearn.cluster

x0 = np.array([ -4.9, -0.03,  6.3, -5.3])

def get_data(seq, start_idx, num_frames):
    basedir = '../fuzzy-metaballs/odometry/'
    sequence_name = str(seq).zfill(2)
    data = pykitti.odometry(basedir, sequence_name, frames=range(start_idx, start_idx + num_frames))
    depths = []
    rays = []
    for t in range(num_frames):
        lidar = np.array(data.get_velo(t), np.float64)
        pose = data.poses[t]
        Tr = data.calib.T_cam0_velo
        Tr_inv = np.linalg.inv(Tr)
        pose = np.matmul(Tr_inv, np.matmul(pose, Tr))
        lidar[:, 3] = 1
        global_lidar = (pose @ lidar.T).T
        center = (pose @ np.array([0, 0, 0, 1]))[:3]
        line = global_lidar[:, :3] - center
        depth = np.sqrt((line ** 2).sum(1))
        ray = line / depth[:, None]

        # filtering:
        bev_depth = np.sqrt(line[:, 0] ** 2 + line[:, 1] ** 2)
        valid = bev_depth > 2.65  # too close to car
        valid2 = lidar[:, 2] > -2.25  # below ground plane
        valid3 = depth < 30  # too far away from sensor
        valid = np.logical_and(valid, valid2)
        valid = np.logical_and(valid, valid3)
        ray = ray[valid]
        depth = depth[valid]

        trans = np.tile(center[None], (ray.shape[0], 1))
        ray = np.stack([ray, trans], 1)
        ray = jnp.array(ray).reshape((-1, 2, 3))
        depth = jnp.array(depth.ravel()).astype(jnp.float32)
        rays.append(ray)
        depths.append(depth)

    rays = jnp.concatenate(rays, 0)
    depths = jnp.concatenate(depths, 0)
    #print('dataset negative percent: ',(depth.ravel() < 0).sum()/depths.ravel().shape[0])
    return depths, rays

def initialize(depth, rays):
    num_gaussians = 20
    # num_gaussians = 1000
    pt_cld = depth[:,None] * rays[:,0] + rays[:,1]

    if False:
        init_index = np.random.choice(len(depth), size=num_gaussians, replace=False)
        init_depth = depth[init_index]
        init_rays = rays[init_index]
        init_means = init_rays[:, 0, :] * init_depth[:, None] + init_rays[:, 1, :]
        init_sphere_size = 1./30
        init_prec = np.array([np.identity(3)/init_sphere_size for _ in range(num_gaussians)])
        init_weights_log = np.log(np.ones(num_gaussians) / num_gaussians)
        init_params = [init_means, init_prec, init_weights_log]
    elif True:
        while True:
            k_means = sklearn.cluster.MiniBatchKMeans(num_gaussians,n_init=3,compute_labels=True)
            k_means.fit(pt_cld)
            if np.unique(k_means.labels_).shape[0] == num_gaussians:
                break
        means = []
        precs = []
        weights = []
        for i in range(num_gaussians):
            pts = np.array(pt_cld)[k_means.labels_ == i]
            means.append(np.mean(pts,0))
            precs.append(np.linalg.cholesky(np.linalg.pinv(np.cov(pts.T))).T)
            weights.append(pts.shape[0])
        weights = np.array(weights) + 1
        weights = weights/weights.sum()
        init_params = [np.array(means), np.array(precs), np.log(weights)]
    else:
        gmm_model = sklearn.mixture.GaussianMixture(num_gaussians,max_iter=3)
        gmm_model.fit(pt_cld)
        init_params = [gmm_model.means_, gmm_model.precisions_cholesky_, np.log(gmm_model.weights_)]
    return init_params


def calc_loss(params, gt_depth):
    means, prec, weights_log, camera_rays, beta2, beta3, beta4, beta5 = params
    render_res = fm_render.render_func_rays(means, prec, weights_log, camera_rays, beta2, beta3, beta4, beta5)
    render_depth = render_res[0]
    avg_depth = gt_depth.mean()


    err = jnp.abs(render_depth - gt_depth)/avg_depth
    depth_loss = err.mean()
    est_alpha = render_res[2]
    est_alpha = jnp.clip(est_alpha,1e-6,1-1e-6)

    mask_loss = -0.5*jnp.log(est_alpha)
    return depth_loss + mask_loss.mean()


def error_func(est_depth,est_alpha,true_depth):
    cond = jnp.isnan(est_depth) | jnp.isnan(true_depth)
    valid_depth_frac =   (~jnp.isnan(cond)).sum()/cond.shape[0]
    avg_depth = jnp.where(cond,0,true_depth).mean()/valid_depth_frac
    err = (est_depth - true_depth)/avg_depth
    depth_loss =  (jnp.where(cond,0,err)**2).mean()

    true_alpha = ~jnp.isnan(true_depth)
    est_alpha = jnp.clip(est_alpha,1e-6,1-1e-6)
    mask_loss = -((true_alpha * jnp.log(est_alpha)) + (~true_alpha)*jnp.log(1-est_alpha))
    
    loss_mul = true_alpha.sum()
    term1 = depth_loss.mean()
    term2 = mask_loss.mean()
    return (term1 + term2)


jit_render = jax.jit(fm_render.render_func)
loss_func = jax.jit(calc_loss)

class SynthFMBFunc:
        def __init__(self,init_params,camera_rays,axangl_true,trans_true,jax_tdepth,obj_scale):
            self.init_params = init_params
            self.camera_rays = camera_rays
            self.axangl_true = axangl_true
            self.trans_true = trans_true
            self.jax_tdepth = jax_tdepth
            self.obj_scale = obj_scale
        def __call__(self,params):
            beta2 = jnp.float32(np.exp(params[0]))
            beta3 = jnp.float32(np.exp(params[1]))
            beta4 = jnp.float32(np.exp(params[2]))
            beta5 = -jnp.float32(np.exp(params[3]))

            render_res = jit_render(self.init_params[0], self.init_params[1], self.init_params[2],
                                    self.camera_rays,self.axangl_true,self.trans_true,
                                    beta2/self.obj_scale,beta3,beta4,beta5)
            err = error_func(render_res[0],render_res[2],self.jax_tdepth)
            return float(err)
        def render(self, params):
            beta2 = jnp.float32(np.exp(params[0]))
            beta3 = jnp.float32(np.exp(params[1]))
            beta4 = jnp.float32(np.exp(params[2]))
            beta5 = -jnp.float32(np.exp(params[3]))

            render_res = jit_render(self.init_params[0], self.init_params[1], self.init_params[2],
                                    self.camera_rays,self.axangl_true,self.trans_true,
                                    beta2/self.obj_scale,beta3,beta4,beta5)
            est_depth = render_res[0]
            est_alpha = render_res[1]
            return est_depth,est_alpha,self.jax_tdepth
         
class KITTIFMBFunc:
        def __init__(self,depths,rays,init_params):
            self.depths = depths
            self.rays = rays
            self.init_params = init_params
        def __call__(self,params):
            sample_N = 10000
            r_idx = np.arange(self.rays.shape[0])
            np.random.shuffle(r_idx)
            r_idx = r_idx[:sample_N]
            beta2 = jnp.float32(np.exp(params[0]))
            beta3 = jnp.float32(np.exp(params[1]))
            beta4 = jnp.float32(np.exp(params[2]))
            beta5 = -jnp.float32(np.exp(params[3]))

            obj_scale = 0.1
            derr = loss_func([self.init_params[0], self.init_params[1], self.init_params[2], self.rays[r_idx], 
                              beta2 / obj_scale, beta3, beta4,beta5], self.depths[r_idx])
            return float(derr)

class FuzzyFunc:
    def __init__(self): 

        import pickle
        eval_data = []
        in_vecs = []
        for i in range(0,200,10):
            depths, rays = get_data(0, i, 10)
            init_params = initialize(depths, rays)
            eval_data.append(init_params)
            in_vecs.append((depths, rays))

        dataset_name = '../fm_render/recon_dataset_1657913714.pkl'
        time_num = dataset_name.split('.')[-2].split('_')[-1]
        with open(dataset_name,'rb') as fp:
            model_set = pickle.load(fp)

        gmm_fold = '../fm_render/fms_{}.pkl'.format(time_num)
        with open(gmm_fold,'rb') as fp:
            good_gmms = pickle.load(fp)
        self.good_gmms = good_gmms
        self.model_set = model_set
        
        f_test = []
        for init_params, in_d in zip(eval_data,in_vecs):
            depths, rays = in_d
            obj = KITTIFMBFunc(depths,rays,init_params)
            f_test.append(obj)
        
        for model_name in model_set:
            model_ex = model_set[model_name]
            mean,prec,weight_log = good_gmms[model_name]

            # load data
            shape_scale = model_ex['scale']
            image_size = model_ex['resolution']

            # should work "better" in the normalized data
            obj_scale = shape_scale/120

            # but in general we don't know so we just use this
            weights = np.exp(weight_log)
            weights = weights/weights.sum()
            obj_scale = (weights[:,None] * mean).std(0).mean()

            model_results = []
            for frame,cpose in zip(model_ex['depth_test'][:2],model_ex['cameras_test'][:2]):
                camera_rays,axangl_true,trans_true = cpose
                jax_tdepth = jnp.array(frame.ravel())
                obj = SynthFMBFunc(good_gmms[model_name],camera_rays,axangl_true,trans_true,jax_tdepth,obj_scale)
                f_test.append(obj)
        
        self.dataset = f_test
        self.balanced_split = np.arange(0,len(f_test),2)
        
        self.train_data = [self.dataset[i] for i in self.balanced_split]
        self.test_data = [self.dataset[i] for i in range(len(self.dataset)) if i not in self.balanced_split]
        
        self.results_log = {}
        self.min_vec = 9e12*np.ones(len(self.balanced_split))
    def reset(self):
        self.results_log = {}
        self.min_vec = 9e12*np.ones(len(self.balanced_split))

    def joint_func(self,x):
        f_res = np.array([test_func(x) for test_func in self.train_data])
        self.min_vec = np.minimum(self.min_vec,f_res)
        self.results_log[tuple(x)] = f_res
        return f_res.mean()
    def idx_func(self,x,idx=[]):
        f_res = np.array([self.train_data[i](x) for i,_ in enumerate(idx) if _])
        self.min_vec[idx] =  np.minimum(self.min_vec[idx],np.array(f_res))
        return f_res