

from settings.defaults import _C
from settings.setup_functions import *
import torch.distributed as dist
import torch.multiprocessing as mp


root = os.path.dirname(os.path.abspath(__file__))
config = _C.clone()
# cfg_file = os.path.join('configs','baseline', 'swin_tiny.yaml')
# cfg_file = os.path.join('../configs', 'eval', 'eval.yaml')
# cfg_file = os.path.join('configs', 'eval', 'eval_base.yaml')
cfg_file = os.path.join('configs', 'swin-cub.yaml')  # visual 需要加前面
config = SetupConfig(config, cfg_file)
config.defrost()
## Log Name and Perferences
config.write = True
config.train.checkpoint = True
config.misc.exp_name = f'{config.data.dataset}'
# config.misc.exp_name = f'cars'
# config.misc.log_name = f'pr {config.parameters.parts_ratio}+pd {config.parameters.parts_drop}'
config.misc.log_name = f'Ours'

# Environment Settings
config.data.log_path = os.path.join(config.misc.output, config.misc.exp_name, config.misc.log_name
                                    + time.strftime(' %m-%d_%H-%M', time.localtime()))

config.model.pretrained = os.path.join(config.model.pretrained,
                                       config.model.name + config.model.pre_version + config.model.pre_suffix)
#
# os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
# os.environ['OMP_NUM_THREADS'] = '1'
#
# # Setup Functions
# config.nprocess, config.local_rank = SetupDevice()
# config.data.data_root, config.data.batch_size = LocateDatasets(config)
# config.train.lr = ScaleLr(config)
# log = SetupLogs(config, config.local_rank)
# if config.write and config.local_rank in [-1, 0]:
# 	with open(config.data.log_path + '/config.json', "w") as f:
# 		f.write(config.dump())
# config.freeze()
# SetSeed(config)

config.distributed = config.distribute.multiprocessing_distributed
ngpus_per_node = torch.cuda.device_count()
config.world_size = ngpus_per_node * config.distribute.world_size

config.train.lr = ScaleLr(config)



