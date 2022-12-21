import torch
from torch import optim
from models.musebert.curricula import all_curriculum, Curriculum

from models.musebert.amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, \
    ConstantScheduler, TrainingInterface
from models.musebert.utils import get_linear_schedule_with_warmup
from typing import Union
from models.musebert.train import TrainMuseBERT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
readme_fn = __file__

clip = 1
parallel = False

# create data_loaders and initialize model specified by the curriculum.
curriculum=all_curriculum
data_loaders, model = curriculum(device)

# load a pre-trained model if necessary.
model_path = '../pretrained/musebert.pt'
model.load_model(model_path, device)

# to handle the path to save model parameters, logs etc.
log_path_mng = LogPathManager(readme_fn)

# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=curriculum.lr['lr'])
schdl_step = len(data_loaders.train_loader) * curriculum.lr['final_epoch']
scheduler = \
    get_linear_schedule_with_warmup(optimizer,
                                    curriculum.lr['warmup'],
                                    schdl_step,
                                    curriculum.lr['final_lr_factor'])
optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, clip)

# tensorboard writers
writer_names = ['loss', 'o_bt', 'o_sub', 'p_hig', 'p_reg',
                'p_deg', 'd_hlf', 'd_sqv']
tags = {'loss': None}
summary_writers = SummaryWriters(writer_names, tags,
                                    log_path_mng.writer_path)

# keyword training parameters
beta_scheduler = ConstantScheduler(curriculum.beta)
params_dic = dict(beta=beta_scheduler)
param_scheduler = ParameterScheduler(**params_dic)

# initialize the training interface
musebert_train = \
    TrainMuseBERT(device, model, parallel, log_path_mng, data_loaders,
                    summary_writers, optimizer_scheduler,
                    param_scheduler, curriculum.lr['n_epoch'])

# start training
musebert_train.run()

# # Input format

# from models.edit_musebert import EditMuseBERT
# device = 'cpu'
# model = EditMuseBERT(device)
# print(model)
