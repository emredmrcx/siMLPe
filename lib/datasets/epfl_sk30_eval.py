import os
import glob
import ast
import numpy as np

import torch
import torch.utils.data as data

class EPFLSK30Eval(data.Dataset):
    def __init__(self, config, split_name):
        super(EPFLSK30Eval, self).__init__()
        self._split_name = split_name
        
        self._epfl_anno_dir = config.epfl_anno_dir
        self._epfl_files = self._get_epfl_files()
        
        self.epfl_motion_input_length = config.motion.epfl_input_length
        self.epfl_motion_target_length = config.motion.epfl_target_length_eval
        
        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step
        self.sample_rate = config.motion.sample_rate
        self._collect_all()
        self._file_length = len(self.data_idx)
    
    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._epfl_files)
    
    def _get_epfl_files(self):
        # Find all pose3d_smpl.csv files
        file_pattern = os.path.join(self._epfl_anno_dir, self._split_name, '*', '*', 'pose_3d', 'pose3d_smpl.csv')
        file_list = glob.glob(file_pattern)
        
        epfl_files = []
        for csv_path in file_list:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            # find kp3ds column index
            header = lines[0].strip().split(',')
            kp3ds_col_idx = header.index('kp3ds')
            
            kp3ds_list = []
            for line in lines[1:]:  # Skip header
                
                # The kp3ds column contains [[...], [...], ...] 
                parts = line.strip().split(',')
                if kp3ds_col_idx == 1:  # kp3ds is the second column
                    # Find the start of kp3ds (after first comma)
                    start_idx = line.find(',') + 1
                    # Find where the closing ]] is
                    end_idx = line.find(']]', start_idx) + 2
                    kp3ds_str = line[start_idx:end_idx].strip()
                    
                    # Remove quotes if present
                    if kp3ds_str.startswith('"'):
                        kp3ds_str = kp3ds_str[1:]
                    if kp3ds_str.endswith('"'):
                        kp3ds_str = kp3ds_str[:-1]
                    
                    kp3d = np.array(ast.literal_eval(kp3ds_str))  # Shape: (17, 3)
                    kp3ds_list.append(kp3d)
            
            # Convert to numpy array and flatten
            kp3ds_array = np.array(kp3ds_list)  # Shape: (T, 17, 3)
            T = kp3ds_array.shape[0]
            kp3ds_flat = kp3ds_array.reshape(T, -1)  # Shape: (T, 51)
            epfl_files.append(torch.from_numpy(kp3ds_flat).float())
        
        return epfl_files
    
    def _collect_all(self):
        # Keep aligned with H36M dataloader approach
        self.epfl_seqs = []
        self.data_idx = []
        idx = 0
        
        for epfl_motion_poses in self._epfl_files:
            N = len(epfl_motion_poses)
            if N < self.epfl_motion_target_length + self.epfl_motion_input_length:
                continue
            
            sample_rate = self.sample_rate
            sampled_index = np.arange(0, N, sample_rate)
            epfl_motion_poses = epfl_motion_poses[sampled_index]
            
            T = epfl_motion_poses.shape[0]
            
            self.epfl_seqs.append(epfl_motion_poses)
            valid_frames = np.arange(0, T - self.epfl_motion_input_length - self.epfl_motion_target_length + 1, self.shift_step)
            
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1
    
    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.epfl_motion_input_length + self.epfl_motion_target_length)
        motion = self.epfl_seqs[idx][frame_indexes]

        # Reshape from (time, 51) to (time, 17, 3)
        motion = motion.reshape(motion.shape[0], 17, 3)
        
        epfl_motion_input = motion[:self.epfl_motion_input_length].float()
        epfl_motion_target = motion[self.epfl_motion_input_length:].float()
        
        return epfl_motion_input, epfl_motion_target