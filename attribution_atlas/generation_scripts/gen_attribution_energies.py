import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
from faccent.utils import get_crop_bounds
from faccent.cam import *
from subprocess import call

rl = torch.nn.ReLU(inplace = False)

def main(cfg_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Import the configuration file as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", cfg_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    out_name = getattr(config, 'out_name', None)

    # Access variables
    attribution_layers = getattr(config, 'attribution_layers', None)  # Provides a default value of None if not found
    target_layer = getattr(config, 'target_layer', None)
    feature = getattr(config, 'feature', None)
    position = getattr(config, 'position', None)
    dataloader = getattr(config, 'dataloader', None)
    DEVICE = getattr(config, 'DEVICE', None)
    model = getattr(config, 'model', None)
    MODEL_INPUT_RANGE = getattr(config, 'MODEL_INPUT_RANGE', None)
    MODEL_INPUT_SIZE = getattr(config, 'MODEL_INPUT_SIZE', None)  

    _ = model.to(DEVICE)
    batch_size = dataloader.batch_size
    total_images = len(dataloader.dataset)

    target_activations = torch.zeros(total_images)
    # Run data through model
    with layer_saver(model, target_layer, detach=False) as target_saver:
        with actgrad_extractor(model, attribution_layers, concat=False) as score_saver:
            for i, data in enumerate(dataloader):
                if i%100 == 0:
                    print(str(i)+'/'+str(len(dataloader)))
                inputs, label = data
                inputs = inputs.to(DEVICE)
                model.zero_grad() # very important!

                batch_layer_activations = target_saver(inputs)[target_layer]
                if len(batch_layer_activations.shape) == 4:
                    if position == 'middle':
                        batch_target_activations = batch_layer_activations[:,:,batch_layer_activations.shape[2]//2,batch_layer_activations.shape[3]//2]
                    elif position is not None:
                        batch_target_activations = batch_layer_activations[:,:,position[0],position[1]]
                    else: batch_target_activations = batch_layer_activations
                else:
                    batch_target_activations = batch_layer_activations
                    

                if isinstance(feature,int):
                    batch_feature_act = batch_target_activations[:,feature]
                else:
                    feature = torch.tensor(feature).float().to(DEVICE)
                    batch_feature_act = torch.matmul(batch_target_activations, torch.tensor(feature).unsqueeze(1)).squeeze()

                loss = torch.sum(batch_feature_act)
                loss.backward()

                # Store results in preallocated space
                target_activations[i*batch_size:(i+1)*batch_size] = batch_feature_act.detach().cpu()

                activations = score_saver.activations
                gradients = score_saver.gradients


                #get shapes
                if i == 0:
                    layer_shapes = {}
                    layer_shapes[target_layer] = batch_layer_activations.shape[1:]
                    attr_crop_bounds = {}
                    for layer in attribution_layers:
                        layer_shapes[layer] = activations[layer].shape[1:]
                        attr_crop_bounds[layer] = find_crop_bounds(torch.abs(gradients[layer]).sum(dim=(0,1)),0.01) #EDIT: this may not work for very sparse inputs!
                    positive_attribution_l1_maps={l: torch.zeros(total_images, attr_crop_bounds[l][0][1]-attr_crop_bounds[l][0][0],attr_crop_bounds[l][1][1]-attr_crop_bounds[l][1][0]) for l in attribution_layers}
                    negative_attribution_l1_maps={l: torch.zeros(total_images, attr_crop_bounds[l][0][1]-attr_crop_bounds[l][0][0],attr_crop_bounds[l][1][1]-attr_crop_bounds[l][1][0]) for l in attribution_layers}
                    positive_attribution_l2_maps={l: torch.zeros(total_images, attr_crop_bounds[l][0][1]-attr_crop_bounds[l][0][0],attr_crop_bounds[l][1][1]-attr_crop_bounds[l][1][0]) for l in attribution_layers}
                    negative_attribution_l2_maps={l: torch.zeros(total_images, attr_crop_bounds[l][0][1]-attr_crop_bounds[l][0][0],attr_crop_bounds[l][1][1]-attr_crop_bounds[l][1][0]) for l in attribution_layers}
                    attribution_l2_maps=         {l: torch.zeros(total_images, attr_crop_bounds[l][0][1]-attr_crop_bounds[l][0][0],attr_crop_bounds[l][1][1]-attr_crop_bounds[l][1][0]) for l in attribution_layers}

                for l in attribution_layers:
                    actgrad = activations[l]*gradients[l]

                    positive_attribution_l1_maps_batch = torch.sum(rl(actgrad), dim=(1))
                    negative_attribution_l1_maps_batch = torch.sum(rl(-actgrad), dim=(1))
                    positive_attribution_l1_maps[l][i*batch_size:(i+1)*batch_size] = positive_attribution_l1_maps_batch[:,attr_crop_bounds[l][0][0]:attr_crop_bounds[l][0][1],attr_crop_bounds[l][1][0]:attr_crop_bounds[l][1][1]]
                    negative_attribution_l1_maps[l][i*batch_size:(i+1)*batch_size] = negative_attribution_l1_maps_batch[:,attr_crop_bounds[l][0][0]:attr_crop_bounds[l][0][1],attr_crop_bounds[l][1][0]:attr_crop_bounds[l][1][1]]
                    
                    positive_attribution_l2_maps_batch = torch.norm(rl(actgrad), dim=1,p=2)
                    negative_attribution_l2_maps_batch = torch.norm(rl(-actgrad), dim=1,p=2)
                    attribution_l2_maps_batch = torch.norm(actgrad, dim=1,p=2)
                    positive_attribution_l2_maps[l][i*batch_size:(i+1)*batch_size] = positive_attribution_l2_maps_batch[:,attr_crop_bounds[l][0][0]:attr_crop_bounds[l][0][1],attr_crop_bounds[l][1][0]:attr_crop_bounds[l][1][1]]
                    negative_attribution_l2_maps[l][i*batch_size:(i+1)*batch_size] = negative_attribution_l2_maps_batch[:,attr_crop_bounds[l][0][0]:attr_crop_bounds[l][0][1],attr_crop_bounds[l][1][0]:attr_crop_bounds[l][1][1]]
                    attribution_l2_maps[l][i*batch_size:(i+1)*batch_size] = attribution_l2_maps_batch[:,attr_crop_bounds[l][0][0]:attr_crop_bounds[l][0][1],attr_crop_bounds[l][1][0]:attr_crop_bounds[l][1][1]]



    # target crop bounds
    recep_field_unit = None
    if isinstance(feature,int):
        recep_field_unit = feature
        
    if len(batch_layer_activations.shape) == 4:
        recep_field, target_crop_bounds = gradient_based_receptive_field(model,target_layer,
                                                                    position=position,
                                                                    unit = recep_field_unit,
                                                                    input_size = MODEL_INPUT_SIZE,
                                                                    input_range= MODEL_INPUT_RANGE,
                                                                    device=DEVICE,
                                                                    integrate = True,
                                                                    init_img = None,
                                                                    crop_threshold = .2,
                                                                    square = True,
                                                                    plot = False)
    else:
        target_crop_bounds = None

    save_obj = {'pos_l1_attr':positive_attribution_l1_maps,
                'neg_l1_attr':negative_attribution_l1_maps,
                'pos_l2_attr':positive_attribution_l2_maps,
                'neg_l2_attr':negative_attribution_l2_maps,
                'l2_attr':attribution_l2_maps,
                'target_activations':target_activations,
                'layer_shapes':layer_shapes,
                'attr_crop_bounds':attr_crop_bounds,
                'target_crop_bounds':target_crop_bounds,
                }
    
    torch.save(save_obj, '../atlases/%s/data/attribution_norms.pt'%(out_name))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <cfg_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)