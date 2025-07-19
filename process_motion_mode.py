import os
import pickle 
import argparse
from utils import get_motion_modes_file
from dataset import PiecewiseDataset
import importlib

parser = argparse.ArgumentParser() 

parser.add_argument('--dataset_path', type=str, default='/depot/bera89/data/zhan5058/TUTR/pedestrain_process')
parser.add_argument('--dataset_name', type=str, default='006')
parser.add_argument("--hp_config", type=str, default=None, help='hyper-parameter')
parser.add_argument('--lr_scaling', action='store_true', default=False)
parser.add_argument('--num_works', type=int, default=8)
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--data_scaling', type=list, default=[1.9, 0.4])
parser.add_argument('--dist_threshold', type=float, default=2)
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("hp_config", args.hp_config)
hp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp_config)

for data_file in os.listdir(args.dataset_path):
    full_data_file = os.path.join(args.dataset_path,data_file)

    train_dataset = PiecewiseDataset(dataset_path=full_data_file, dataset_name=args.dataset_name,
                                    dataset_type='train', translation=True, rotation=True, 
                                    scaling=True, obs_len=args.obs_len, 
                                    dist_threshold=hp_config.dist_threshold, smooth=False) 
    save_folder = "/depot/bera89/data/zhan5058/TUTR/locovr_motion_modes"

    motion_modes = get_motion_modes_file(train_dataset, save_folder, args.obs_len, args.pred_len, hp_config.n_clusters, args.dataset_path, args.dataset_name,
                                        smooth_size=hp_config.smooth_size, random_rotation=hp_config.random_rotation, traj_seg=hp_config.traj_seg)
    print(motion_modes.shape)
    

