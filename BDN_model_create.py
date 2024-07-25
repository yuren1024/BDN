import torch
from utils import network_parameters
from model.base_control_net import  BDCNet

## Build BDCNet
print('Build the model')

model= BDCNet()
p_number = network_parameters(model)
print(p_number)
model.cuda()

model_path = "checkpoints/your/model/name"
pretrained_weights = torch.load(model_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']
target_dict = {}

scratch_dict = model.state_dict()

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

for k in scratch_dict.keys():
    is_control,name = get_node_name(k,'control_model.')
    if is_control:
        copy_k = 'module.' + name
    else:
        copy_k = k
    
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict,strict=True)
torch.save(model.state_dict(),'checkpoints/BDCNet_model.pth')
print("Done.")