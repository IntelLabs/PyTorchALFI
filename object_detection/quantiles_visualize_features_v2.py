import os, sys
sys.path.append("..")
sys.path.append('/home/fgeissle/ranger_repo/ranger/yolov3_ultra')
from yolov3_ultra.yolov3 import yolov3
from os.path import dirname as up
sys.path.append(up(up(up(os.getcwd()))))
from PIL import Image, TiffImagePlugin
TiffImagePlugin.DEBUG = False
from min_test import build_objdet_native_model, parse_opt
import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_model_summary import summary
import math
from obj_det_evaluate_jsons import read_fault_file, load_json_indiv

def load_feature_details(target):
    load_path = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/exp_1_features_' + target + '.json'
    fdict = load_json_indiv(load_path)
    feature_importance = fdict['feature_importance']
    feature_labels = fdict['labels_sorted']
    return feature_importance, feature_labels

# Retrieve layer names -----------------------------------------------
dataset_name = 'coco2017'
opt = parse_opt(dataset_name)
    
# device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() else "cpu")    
yolov3_model = yolov3(**vars(opt))
# opt.device = device

# yolov3_model = yolov3_model.to(device)
model = build_objdet_native_model(model=yolov3_model, opt=opt, dataset_name=dataset_name)

lay_list = {}
hook_list = {}
cnt = 0
for name, m in model.named_modules():
    
    lay_list[str(cnt)] =  name
    if type(m) in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Sigmoid, torch.nn.modules.activation.SiLU]:
        hook_list[str(cnt)] =  name

    cnt += 1
        
n = 6
item = list(hook_list.items())[n] #layer nr in full model, name
lay_list[item[0]]
# hook_layer_names = list(hook_list.values())
# hook_layer_names[n]




# show input shape
# print(summary(model, torch.zeros((1, 3, 100, 100)), show_input=True))
# show output shape
# print(summary(model, torch.zeros((1, 3, 100, 100)), show_input=False))
# show output shape and hierarchical view of net
# print(summary(model, torch.zeros((1, 3, 100, 100)), show_input=False, show_hierarchical=True))




# Plot feature importance  -----------------------------------------------

target = 'neurons_weights' #noise, blur, neurons, weights, blur_noise, neurons_weights


feature_importance, feature_labels = load_feature_details(target)

plt_name = '/home/fgeissle/ranger_repo/ranger/object_detection/quantile_detection_data/feature_importance/feature_imp_ranking_' + target + '.png'

q10_list = {'x': [], 'y': [], 'weight': []}
q25_list = {'x': [], 'y': [], 'weight': []}
q50_list = {'x': [], 'y': [], 'weight': []}
q75_list = {'x': [], 'y': [], 'weight': []}
q100_list = {'x': [], 'y': [], 'weight': []}

rel_feat_imp = (feature_importance/np.sum(feature_importance)).tolist()
for x in range(len(feature_labels)):
    if 'q10_' in feature_labels[x]:
        q10_list['x'].append(int(feature_labels[x][7:])) #x as layer info
        q10_list['y'].append(x) #ranking no
        q10_list['weight'].append(rel_feat_imp[x])
    elif 'q25_' in feature_labels[x]:
        q25_list['x'].append(int(feature_labels[x][7:])) #x as layer info
        q25_list['y'].append(x) #ranking no
        q25_list['weight'].append(rel_feat_imp[x])
    elif 'q50_' in feature_labels[x]:
        q50_list['x'].append(int(feature_labels[x][7:])) #x as layer info
        q50_list['y'].append(x) #ranking no
        q50_list['weight'].append(rel_feat_imp[x])
    elif 'q75_' in feature_labels[x]:
        q75_list['x'].append(int(feature_labels[x][7:])) #x as layer info
        q75_list['y'].append(x) #ranking no
        q75_list['weight'].append(rel_feat_imp[x])
    elif 'q100_' in feature_labels[x]:
        q100_list['x'].append(int(feature_labels[x][8:])) #x as layer info
        q100_list['y'].append(x) #ranking no
        q100_list['weight'].append(rel_feat_imp[x])


#bring all in layer-order
def sort_by_xlist(q10_list):
    x_sorted = [x for x,_,_ in sorted(zip(q10_list['x'],q10_list['y'], q10_list['weight']))]
    y_sorted = [x for _,x,_ in sorted(zip(q10_list['x'],q10_list['y'], q10_list['weight']))]
    w_sorted = [x for _,_,x, in sorted(zip(q10_list['x'],q10_list['y'], q10_list['weight']))]

    q10_list['x'] = x_sorted
    q10_list['y'] = y_sorted
    q10_list['weight'] = w_sorted

    return q10_list

q10_list = sort_by_xlist(q10_list)
q25_list = sort_by_xlist(q25_list)
q50_list = sort_by_xlist(q50_list)
q75_list = sort_by_xlist(q75_list)
q100_list = sort_by_xlist(q100_list)

fig, ax = plt.subplots()
width = 1.0
ax.bar(q10_list['x'], q10_list['weight'], width, label='q10')
ax.bar(q25_list['x'], q25_list['weight'], width, label='q25', bottom=q10_list['weight'])
ax.bar(q50_list['x'], q50_list['weight'], width, label='q50', bottom=(np.array(q10_list['weight'])+np.array(q25_list['weight'])).tolist())
ax.bar(q75_list['x'], q75_list['weight'], width, label='q75', bottom=(np.array(q10_list['weight'])+np.array(q25_list['weight'])+np.array(q50_list['weight'])).tolist())
ax.bar(q100_list['x'], q100_list['weight'], width, label='q100', bottom=(np.array(q10_list['weight'])+np.array(q25_list['weight'])+np.array(q50_list['weight'])+np.array(q75_list['weight'])).tolist())

# Plot labels
fnt_size = 10
for u in range(len(rel_feat_imp)):
    if rel_feat_imp[u] < 0.1:
        break
    if 'q100_' in feature_labels[u]:
        lbl = feature_labels[u][8:]
    else:
        lbl = feature_labels[u][7:]
    ax.text(int(lbl)-0.05, rel_feat_imp[u] +0.05, lbl, fontsize=fnt_size-1)  # , color='blue', fontweight='bold')
    print('lay', int(lbl), 'rel imp', rel_feat_imp[u])


plt.ylim([0,1])
plt.xlim([-0.5,71.5])
plt.legend()
plt.xlabel('Layer')
plt.ylabel('Relative feature importance')
# plt.xticks(fontsize=4)
plt.savefig(plt_name)

print('Fig saved as ' + plt_name)