import sys
import os
import torch
from subprocess import call
import subprocess
import numpy as np


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
    featureviz_batch_size = getattr(config, 'featureviz_batch_size', None)
    icon_size = getattr(config, 'icon_size', None)


    for attr_layer in attribution_layers:
        icons = torch.load('../atlases/%s/data/%s_atlas.pt'%(out_name,attr_layer))['icons']
        directions = torch.tensor(np.stack(icons[:,0]))
        feature_acts = torch.zeros(directions.shape[0])
        torch.save(feature_acts,'../atlases/%s/data/%s_icon_acts.pt'%(out_name,attr_layer))
        icon_imgs = torch.zeros(directions.shape[0],3,icon_size,icon_size)
        icon_img_trs = torch.zeros(directions.shape[0],icon_size,icon_size)
        torch.save(icon_imgs,'../atlases/%s/data/%s_icon_imgs.pt'%(out_name,attr_layer))
        icon_img_trs = np.zeros((directions.shape[0],icon_size,icon_size))
        torch.save(icon_img_trs,'../atlases/%s/data/%s_icon_img_trs.pt'%(out_name,attr_layer))

        for batch_i,_ in enumerate(range(0, directions.shape[0], featureviz_batch_size)):
            with open('../logs/%s.out'%out_name, 'a') as log_file:
                subprocess.run(['python', 
                                'gen_batch_visualization.py',
                                cfg_path, 
                                attr_layer,
                                str(batch_i)],
                            stdout=log_file,  
                            stderr=log_file
                            ) 


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <cfg_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)
    