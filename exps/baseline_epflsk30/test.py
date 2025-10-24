import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config  import config
from model import siMLPe as Model
from datasets.epfl_sk30_eval import EPFLSK30Eval

import torch
from torch.utils.data import DataLoader

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

# Joint names for EPFL-SK30 dataset
joint_names = [
    'Root',
    'LeftHead',
    'RightHead',
    'LeftBody',
    'RightBody',
    'LeftShoulder',
    'RightShoulder',
    'LeftArm',
    'RightArm',
    'LeftForearm',
    'RightForearm',
    'LeftHip',
    'RightHip',
    'LeftKnee',
    'RightKnee',
    'LeftFoot',
    'RightFoot'
]

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.epfl_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def regress_pred(model, pbar, num_samples, m_p3d_h36_per_joint):

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c,_ = motion_input.shape
        num_samples += b

        # Reshape to (b, n, 51)
        motion_input = motion_input.reshape(b, n, 17, 3)
        motion_input = motion_input.reshape(b, n, -1)
        outputs = []
        
        # step = training output length
        # autoregressively predict poses 
        step = config.motion.epfl_target_length_train  
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.epfl_input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.epfl_input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(-1, 17*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            
            # Update input: remove first 'step_length' poses and append predicted poses
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape

        motion_gt = motion_target.clone()

        motion_pred = motion_pred.detach().cpu()
        motion_pred = motion_pred.clone().reshape(b,n,17,3)

        # Calculate MPJPE per joint for each timestep
        # dim=3: L2 norm over (x,y,z) -> (b, n, 17)
        # dim=0: sum over batch -> (n, 17)
        mpjpe_p3d_h36_per_joint = torch.sum(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=0)
        m_p3d_h36_per_joint += mpjpe_p3d_h36_per_joint.cpu().numpy()
    
    m_p3d_h36_per_joint = m_p3d_h36_per_joint / num_samples
    return m_p3d_h36_per_joint

def test(config, model, dataloader, return_per_joint=False):

    m_p3d_h36_per_joint = np.zeros([config.motion.epfl_target_length_eval, 17])  # 17 joints
    titles = np.array(range(config.motion.epfl_target_length_eval)) + 1
    num_samples = 0

    pbar = dataloader
    m_p3d_h36_per_joint = regress_pred(model, pbar, num_samples, m_p3d_h36_per_joint)
    
    # Calculate mean over joints
    m_p3d_h36 = m_p3d_h36_per_joint.mean(axis=1)

    ret = {}
    ret_per_joint = {}
    for j in range(config.motion.epfl_target_length_eval):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
        ret_per_joint["#{:d}".format(titles[j])] = m_p3d_h36_per_joint[j]
    
    if return_per_joint:
        return [round(ret[key][0], 1) for key in results_keys], ret_per_joint
    else:
        return [round(ret[key][0], 1) for key in results_keys]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    parser.add_argument('--show-per-joint', action='store_true', help='Show per-joint errors')
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    

        
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    model.cuda()

    config.motion.epfl_target_length = config.motion.epfl_target_length_eval
    dataset = EPFLSK30Eval(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    if args.show_per_joint:
        acc_tmp, ret_per_joint = test(config, model, dataloader, return_per_joint=True)
        print("Overall Results:")
        print(results_keys)
        print(acc_tmp)
        
        print("\n" + "="*60)
        print("Per-Joint Errors for each timestep:")
        print("="*60)
        for key in results_keys:
            print(f"\nTimestep {key}:")
            per_joint_errors = ret_per_joint[key]
            for joint_idx in range(len(per_joint_errors)):
                print(f"  Joint {joint_idx:2d} ({joint_names[joint_idx]:13s}): {per_joint_errors[joint_idx]:.3f} mm")
            print(f"  Average: {acc_tmp[results_keys.index(key)]:.1f} mm")
    else:
        print(test(config, model, dataloader))

