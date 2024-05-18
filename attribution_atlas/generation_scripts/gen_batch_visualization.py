import sys
import os
import torch
from subprocess import call
import subprocess
import numpy as np
from faccent import param, transform, render, objectives
from faccent.param import fourier_phase
from faccent.cam import layer_saver
import torch.nn.functional as F
from faccent.utils import scale_crop_bounds
from faccent.utils import clip_percentile, gaussian_filter, torch_to_numpy
from scipy.ndimage import zoom




def prep_img_trs(img_trs, p, blur_sigma=2.0):
    img_trs = torch_to_numpy(img_trs)
    new_trs = []
    n = img_trs.shape[0]
    for i in range(n):
        curr_tr = img_trs[i]

        if curr_tr.shape[0] == 1:
            curr_tr = np.moveaxis(curr_tr, 0, -1)


        curr_tr = np.mean(np.array(curr_tr).copy(), -1, keepdims=True)
        curr_tr = clip_percentile(curr_tr, p)
        curr_tr = curr_tr / curr_tr.max()

        # Blur transparency
        curr_tr = curr_tr.squeeze()
        curr_tr = gaussian_filter(curr_tr, sigma=blur_sigma)
        curr_tr = curr_tr[:, :]
        
        new_trs.append(curr_tr)
        
        
    return np.stack(new_trs)

def interpolate_and_clip(array, new_shape):
    # array: Input array of shape [b, h, w]
    # new_shape: Tuple of new dimensions (h, w)
    
    # Calculate the zoom factors for the h and w dimensions
    zoom_factors = [new_shape[0] / array.shape[1], new_shape[1] / array.shape[2]]
    
    # Initialize an empty list to hold the interpolated arrays
    output = []
    
    # Interpolate each batch element individually
    for i in range(array.shape[0]):
        # Apply zoom to the ith element of the batch. Note: No zoom factor for the batch dimension
        interpolated = zoom(array[i], zoom_factors, order=3)  # Cubic interpolation for better quality
        # Append the interpolated image to the output list
        output.append(interpolated)
    
    # Convert the list of arrays back into a single numpy array
    output_array = np.array(output)
    
    # Clip the values of the output array to be between 0 and 1
    output_clipped = np.clip(output_array, 0, 1)
    
    return output_clipped




def main(cfg_path, attr_layer, batch_i):

    print('batch '+str(batch_i))
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
    canvas_size = getattr(config, 'canvas_size', None)
    featureviz_batch_size = getattr(config, 'featureviz_batch_size', None)
    out_thresholds = getattr(config, 'out_thresholds', None)
    copy_batch = getattr(config, 'copy_batch', None)
    seed = getattr(config, 'seed', None)
    cosine_power = getattr(config, 'cosine_power', None)
    optimizer = getattr(config, 'optimizer', None)
    negative_boost = getattr(config, 'negative_boost', None)
    DEVICE = getattr(config, 'DEVICE', None)
    model = getattr(config, 'model', None)
    simple_preprocess = getattr(config, 'simple_preprocess', None)
    feature = getattr(config, 'feature', None)
    MODEL_INPUT_SIZE = getattr(config, 'MODEL_INPUT_SIZE', None)
    icon_size = getattr(config, 'icon_size', None)
    position = getattr(config, 'position', None)
    crop_adjustment = getattr(config, 'crop_adjustment', None)
    trans_p = getattr(config, 'trans_p', None)
    
    if not isinstance(feature,int):
        feature = feature.to(DEVICE)

    _ = model.to(DEVICE)

    #previously generated data
    icons = torch.load('../atlases/%s/data/%s_atlas.pt'%(out_name,attr_layer))['icons']
    norms_data = torch.load('../atlases/%s/data/attribution_norms.pt'%out_name)
    attr_crop_bounds = norms_data['attr_crop_bounds'][attr_layer]
    target_crop_bounds = norms_data['target_crop_bounds']
    layer_shapes = norms_data['layer_shapes']
    attr_crop_shape = torch.tensor(attr_crop_bounds)[:,1]-torch.tensor(attr_crop_bounds)[:,0] 
    
    scaled_crop_bounds = None
    if target_crop_bounds is not None:
        scaled_crop_bounds = scale_crop_bounds(target_crop_bounds,MODEL_INPUT_SIZE[0],canvas_size[0])
        scaled_crop_bounds = [[scaled_crop_bounds[0][0]-crop_adjustment,scaled_crop_bounds[0][1]+crop_adjustment],
                             [scaled_crop_bounds[1][0]-crop_adjustment,scaled_crop_bounds[1][1]+crop_adjustment]]
        print(scaled_crop_bounds)

    all_feature_acts = torch.load('../atlases/%s/data/%s_icon_acts.pt'%(out_name,attr_layer))
    all_icon_imgs = torch.load('../atlases/%s/data/%s_icon_imgs.pt'%(out_name,attr_layer))
    all_icon_img_trs = torch.load('../atlases/%s/data/%s_icon_img_trs.pt'%(out_name,attr_layer))

    directions = torch.tensor(np.stack(icons[:,0]))
    direction_norms = directions.view(directions.size(0), -1).norm(p=2, dim=1, keepdim=True)
    directions = directions/direction_norms.view(-1, 1, 1, 1)
    target_maps = torch.zeros(directions.shape[0],directions.shape[1],layer_shapes[attr_layer][-2],layer_shapes[attr_layer][-1])
    target_maps[:,:,attr_crop_bounds[0][0]:attr_crop_bounds[0][1],attr_crop_bounds[1][0]:attr_crop_bounds[1][1]] = directions

    sel_target_maps = target_maps[batch_i*featureviz_batch_size:(batch_i+1)*featureviz_batch_size]
    negative_mask = sel_target_maps < 0
    sel_target_maps[negative_mask] *= negative_boost
    sel_target_maps = torch.abs(sel_target_maps)

    print('target map shape')
    print(sel_target_maps.shape)

    parameterizer = fourier_phase(
                                        device=DEVICE,
                                        img_size = canvas_size,
                                        batch_size = sel_target_maps.shape[0],
                                        seed=seed,
                                        copy_batch = copy_batch
                                        )


    for i in range(sel_target_maps.shape[0]):
        if i == 0:
            obj = objectives.spatial_channel_cosim(attr_layer,
                                                sel_target_maps[i],
                                                 relu_act=False, 
                                                 cosine_power = cosine_power,
                                                 crop_bounds = attr_crop_bounds,
                                                 batch=i)
        else:
            obj += objectives.spatial_channel_cosim(attr_layer,
                                                sel_target_maps[i],
                                                 relu_act=False, 
                                                 cosine_power = cosine_power,
                                                 crop_bounds = attr_crop_bounds,
                                                 batch=i)


    transforms = [transform.box_crop(box_min_size=0.9,
                                         box_max_size=0.99,
                                         box_loc_std=0.05,
                        ),
                transform.uniform_gaussian_noise()
                     ]

    imgs, img_trs, losses, _ = render.render_vis(model,
                                                obj,
                                                parameterizer = parameterizer,
                                                transforms = transforms,
                                                optimizer = optimizer,
                                                nb_transforms = 1,
                                                #img_tr_obj = obj1,
                                                img_size = canvas_size,
                                                out_thresholds = out_thresholds,
                                                inline_thresholds = [],
                                                trans_p= 3)

    processed_imgs = simple_preprocess(torch.concatenate(imgs)).to(DEVICE)
    print('processed image shape:')
    print(processed_imgs.shape)
    with layer_saver(model, target_layer, detach=True) as target_saver:
        layer_acts = target_saver(processed_imgs)[target_layer]


    if isinstance(feature,int):
        feature_acts = layer_acts[:,feature]
    else:
        try:

            feature_acts = torch.sum(layer_acts*feature.reshape(1, -1, 1, 1),dim=1)
        except:
            feature_acts = torch.matmul(layer_acts, torch.tensor(feature).unsqueeze(1)).squeeze()

    print('feature acts shape')
    print(feature_acts.shape)
    if len(feature_acts.shape) == 3:
        if position == 'middle':
            feature_acts = feature_acts[:,feature_acts.shape[-2]//2,feature_acts.shape[-1]//2]
        elif position is not None:
            feature_acts = feature_acts[:,position[0],position[1]]
            
         
    #image trs
    img_trs = prep_img_trs(img_trs[-1].detach().cpu(), trans_p, blur_sigma=2.0)
    print(img_trs.shape)
        
    if scaled_crop_bounds is not None:
        imgs = imgs[-1][:,:,scaled_crop_bounds[0][0]:scaled_crop_bounds[0][1],scaled_crop_bounds[1][0]:scaled_crop_bounds[1][1]].detach().cpu()
        img_trs = img_trs[:,scaled_crop_bounds[0][0]:scaled_crop_bounds[0][1],scaled_crop_bounds[1][0]:scaled_crop_bounds[1][1]]
    else:
        imgs = imgs[-1].detach().cpu()

    imgs = F.interpolate(imgs, size=(icon_size,icon_size), mode='bilinear', align_corners=False)
    img_trs = interpolate_and_clip(img_trs, (icon_size,icon_size))

    all_feature_acts[batch_i*featureviz_batch_size:(batch_i+1)*featureviz_batch_size] = feature_acts
    all_icon_imgs[batch_i*featureviz_batch_size:(batch_i+1)*featureviz_batch_size] = imgs
    all_icon_img_trs[batch_i*featureviz_batch_size:(batch_i+1)*featureviz_batch_size] = img_trs
    torch.save(all_feature_acts,'../atlases/%s/data/%s_icon_acts.pt'%(out_name,attr_layer))
    torch.save(all_icon_imgs,'../atlases/%s/data/%s_icon_imgs.pt'%(out_name,attr_layer))
    torch.save(all_icon_img_trs,'../atlases/%s/data/%s_icon_img_trs.pt'%(out_name,attr_layer))
        

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python script.py <cfg_path>  <attr layer>  <batch_i>")
        sys.exit(1)

    config_path = sys.argv[1]
    attr_layer = sys.argv[2]
    batch_i = int(sys.argv[3])
    main(config_path, attr_layer, batch_i)
    