import pickle as pkl 
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
model_name = 'resnet'
algo = {}
algo['$\lambda$-equitune'] = {}
algo['equitune'] = {}

algo['equizero (with entropy)'] = {}
algo['equizero (with max probability)'] = {}
colors = {}
colors['equitune'] = '#e41a1c'
colors['$\lambda$-equitune'] = '#377eb8'
colors['equizero (with entropy)'] = '#4daf4a'
colors['equizero (with max probability)'] = '#984ea3'

markers = {}
markers['equitune'] = 'v'
markers['$\lambda$-equitune'] = 'o'
markers['equizero (with entropy)'] = 'h'
markers['equizero (with max probability)'] = 'H'

for key in algo.keys():
    algo[key]['mean'] = []
    algo[key]['std'] = []
    algo[key]['hist'] = []

for seed in [0, 1, 2]:
    model_type = 'equi0'
    eval_type = 'equi0'
    use_ori_equizero = False
    use_entropy = False

    filename = './logs/' + model_name + '_' + model_type + '_' + eval_type + '_' + str(use_ori_equizero) + '_' + str(use_entropy) + '_' + str(seed) + '.pkl'
    data = pkl.load(open(filename, 'rb'))
    out = data[0]
    out = np.array([i.cpu().item() for i in out])
    algo['$\lambda$-equitune']['hist'].append(out)

    model_type = 'equitune'
    eval_type = 'equitune'
    use_ori_equizero = False
    use_entropy = False

    filename = './logs/' + model_name + '_' + model_type + '_' + eval_type + '_' + str(use_ori_equizero) + '_' + str(use_entropy) + '_' + str(seed) + '.pkl'
    data = pkl.load(open(filename, 'rb'))
    out = data[0]
    out = np.array([i.cpu().item() for i in out])
    algo['equitune']['hist'].append(out)
    
    
    model_type = 'equi0'
    eval_type = 'equi0'
    use_ori_equizero = True
    use_entropy = False

    filename = './logs/' + model_name + '_' + model_type + '_' + eval_type + '_' + str(use_ori_equizero) + '_' + str(use_entropy) + '_' + str(seed) + '.pkl'
    data = pkl.load(open(filename, 'rb'))
    out = data[0]
    out = np.array([i.cpu().item() for i in out])
    algo['equizero (with entropy)']['hist'].append(out)

    model_type = 'equi0'
    eval_type = 'equi0'
    use_ori_equizero = True
    use_entropy = True

    filename = './logs/' + model_name + '_' + model_type + '_' + eval_type + '_' + str(use_ori_equizero) + '_' + str(use_entropy) + '_' + str(seed) + '.pkl'
    data = pkl.load(open(filename, 'rb'))
    out = data[0]
    out = np.array([i.cpu().item() for i in out])
    algo['equizero (with max probability)']['hist'].append(out)


for key in algo.keys():
    array = np.stack(algo[key]['hist'], axis=0)
    print (np.shape(array))
    x = np.array(range(10))
    algo[key]['mean'] = np.mean(array, axis=0)
    algo[key]['std'] = np.std(array, axis=0)
    plt.plot(x, algo[key]['mean'], label=key, color=colors[key], marker=markers[key])
    plt.fill_between(x, algo[key]['mean'] - algo[key]['std'], algo[key]['mean'] + algo[key]['std'], color=colors[key], alpha=0.1)

plt.grid(True)
plt.legend(prop={'size': 12}, loc='lower right')
plt.xlabel('Epoch', fontsize=15)   
plt.ylabel('Accuracy for Rot$90^{\circ}$-CIFAR10', fontsize=15)
plt.ylim([0.38, 0.82])
plt.xticks(x)
plt.tight_layout()

plt.savefig(model_name + '_finetune' + '.png', dpi=250)
plt.close()





