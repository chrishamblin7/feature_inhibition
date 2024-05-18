import sys
import os
import torch
from subprocess import call
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def add_colored_border(img, value, vmin, vmax, border_size=10):
    """
    Adds a colored border to an image based on a specific value.

    Parameters:
    - img: Input image as a numpy array with values between [0, 1].
    - value: Value used to determine the color of the border.
    - vmin, vmax: Minimum and maximum values used for normalization.
    - border_size: Size of the border in pixels.

    Returns:
    - Image with colored border.
    """
    # Define the colormap and normalization
    cmap = plt.get_cmap('RdBu_r')
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    
    # Determine the color for the border
    border_color = cmap(norm(value))
    
    # Convert the color to RGB format as expected by matplotlib
    border_color_rgb = np.array(border_color[:3])
    
    # Create a new image with a border
    h, w = img.shape[:2]
    new_img = np.zeros((h + 2 * border_size, w + 2 * border_size, 3), dtype=np.uint8)
    
    # Fill the new image with the border color
    new_img[:, :] = border_color_rgb
    
    # Place the original image in the center
    new_img[border_size:border_size+h, border_size:border_size+w] = img
    
    return new_img


def overlay_colored_border(img, value, vmin, vmax, vcenter=0, border_size=5):
    """
    Overlays a colored border on an image based on a specific value, without increasing
    the image size. Modifies the outermost pixels to the border color.

    Parameters:
    - img: Input image as a numpy array with values between [0, 1].
    - value: Value used to determine the color of the border.
    - vmin, vmax: Minimum and maximum values used for normalization.
    - border_size: Thickness of the border in pixels.

    Returns:
    - Image with an overlaid colored border, with values in range [0, 1].
    """
    # Ensure border_size is not larger than half the image dimensions
    h, w = img.shape[:2]
    border_size = min(border_size, h // 2, w // 2)

    # Define the colormap and normalization
    cmap = plt.get_cmap('RdBu_r')
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    
    # Determine the color for the border in the range [0, 1]
    border_color = cmap(norm(value))[:3]  # Exclude alpha value if present
    
    # Overlay the border on the image
    # Top and bottom
    img[:border_size, :] = border_color
    img[-border_size:, :] = border_color
    # Left and right
    img[:, :border_size] = border_color
    img[:, -border_size:] = border_color
    
    return img




def main(cfg_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Import the configuration file as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", cfg_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    out_name = getattr(config, 'out_name', None)
    icon_size = getattr(config, 'icon_size', 80)
    grid_size = getattr(config, 'grid_size', None)
    act_color_cutoff = getattr(config, 'act_color_cutoff', [4,4])
    dpi = getattr(config, 'dpi', 300)
    center_activations = getattr(config, 'center_activations', False)

    # Access variables
    attribution_layers = getattr(config, 'attribution_layers', None)  # Provides a default value of None if not found
    target_layer = getattr(config, 'target_layer', None)

    norms_data = torch.load('../atlases/%s/data/attribution_norms.pt'%out_name)
    target_activations = norms_data['target_activations']
    if center_activations:
        target_activations = target_activations-target_activations.mean()


    vmin = target_activations.mean()-act_color_cutoff[0]*target_activations.std()  # or any specific minimum value you wish
    vmax = target_activations.mean()+act_color_cutoff[1]*target_activations.std()  # or any specific maximum value you wish

    for attr_layer in attribution_layers:
        icons = torch.load('../atlases/%s/data/%s_atlas.pt'%(out_name,attr_layer))['icons']
        icon_acts = torch.load('../atlases/%s/data/%s_icon_acts.pt'%(out_name,attr_layer))
        icon_imgs = torch.load('../atlases/%s/data/%s_icon_imgs.pt'%(out_name,attr_layer))
        icon_img_trs = torch.load('../atlases/%s/data/%s_icon_img_trs.pt'%(out_name,attr_layer))


        canvas = np.ones((icon_size * grid_size[0], icon_size * grid_size[1], 3))
        for i, icon in enumerate(icon_imgs):
            icon_img = icon.numpy().transpose(1, 2, 0)
            #add tr
            tr_repeat = np.repeat(icon_img_trs[i][:, :, np.newaxis],3,axis=2)
            icon_img = tr_repeat * icon_img + (1 - tr_repeat)
            icon_img = overlay_colored_border(icon_img, icon_acts[i], vmin, vmax)
            y = int(icons[i, 1])
            x = int(icons[i, 2])
            canvas[(grid_size[0] - x - 1) * icon_size:(grid_size[0] - x) * icon_size, (y) * icon_size:(y + 1) * icon_size] = icon_img


        plt.imshow(canvas)
        plt.axis('off')
        plt.savefig('../atlases/%s/figures/%s_atlas.png'%(out_name,attr_layer), bbox_inches='tight', dpi=dpi)
        plt.close()
        plt.clf()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <cfg_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)
    