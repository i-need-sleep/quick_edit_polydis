import torch
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data = torch.load('./prec_recall_f1.hi')

plot = sns.displot(data['prec'])
fig = plot.get_figure()
fig.savefig('fig.png')
print('hi')