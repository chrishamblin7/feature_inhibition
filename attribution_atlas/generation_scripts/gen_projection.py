import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
from faccent.utils import get_crop_bounds
from torch.utils.data import Dataset, DataLoader
from faccent.cam import *
from subprocess import call
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import matplotlib
matplotlib.use('Agg') 

rl = torch.nn.ReLU(inplace = False)

def create_sample_condition(d, c):
    # Step 1: Sort x and get the sorting indices
    sorted_indices = torch.argsort(d)
    # Initialize a boolean mask with False
    mask = torch.zeros_like(d, dtype=torch.bool)
    
    # Step 2: Mark the indices we are interested in as True in the sorted mask
    # Note: We use the sorted indices to create a mask in the sorted order
    for idx in c:
        mask[sorted_indices[idx]] = True
    
    return mask


def load_select_indices_in_batches(sel_indices, batch_size, image_folder, all_images, preprocess, device):
    # Function to yield batches of images
    for start_idx in range(0, len(sel_indices), batch_size):
        end_idx = start_idx + batch_size
        img_batch = []
        for i in sel_indices[start_idx:end_idx]:
            img_path = image_folder + all_images[i]
            img = preprocess(Image.open(img_path))
            img_batch.append(img)
        img_batch = torch.stack(img_batch).to(device)
        yield img_batch
        
        
class SelectImageDataset(Dataset):
    def __init__(self, image_folder, all_images, sel_indices, transform=None):
        self.image_folder = image_folder
        self.all_images = [all_images[i] for i in sel_indices]
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.all_images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image
    

def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""

    # compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))

    # add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)

    # embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)

    return clipped

def grid(xpts=None, ypts=None, grid_size=(10,10), x_extent=(0., 1.), y_extent=(0., 1.)):
    xpx_length = grid_size[0]
    ypx_length = grid_size[1]

    xpt_extent = x_extent
    ypt_extent = y_extent

    xpt_length = xpt_extent[1] - xpt_extent[0]
    ypt_length = ypt_extent[1] - ypt_extent[0]

    xpxs = ((xpts - xpt_extent[0]) / xpt_length) * xpx_length
    ypxs = ((ypts - ypt_extent[0]) / ypt_length) * ypx_length

    ix_s = range(grid_size[0])
    iy_s = range(grid_size[1])
    xs = []
    for xi in ix_s:
        ys = []
        for yi in iy_s:
            xpx_extent = (xi, (xi + 1))
            ypx_extent = (yi, (yi + 1))

            in_bounds_x = np.logical_and(xpx_extent[0] <= xpxs, xpxs <= xpx_extent[1])
            in_bounds_y = np.logical_and(ypx_extent[0] <= ypxs, ypxs <= ypx_extent[1])
            in_bounds = np.logical_and(in_bounds_x, in_bounds_y)

            in_bounds_indices = np.where(in_bounds)[0]
            ys.append(in_bounds_indices)
        xs.append(ys)
    return np.asarray(xs)


# Load the RdBu colormap
original_cmap = plt.cm.RdBu_r

# Get the colormap colors and make a copy
colormap_array = original_cmap(np.arange(original_cmap.N))

# Modify the center color to be darker
# Assuming the center is at 0.5 (in the middle of the colormap)
center_idx = int(original_cmap.N * 0.5)  # Index of the center color
darker_grey = np.array([0.9, 0.9, 0.9, 1.0])  # New darker grey color

# Adjust the colors near the center to gradually change to the darker grey
gradient_span = 50  # Span of indices around the center to modify for a smooth transition
for i in range(max(0, center_idx - gradient_span), min(center_idx + gradient_span, original_cmap.N)):
    # Calculate a weight for blending based on the distance from the center
    distance_from_center = abs(i - center_idx) / gradient_span
    colormap_array[i] = (1 - distance_from_center) * darker_grey + distance_from_center * colormap_array[i]

# Create a new colormap from the modified array
modified_rdbu = mcolors.LinearSegmentedColormap.from_list("RdBu_modified", colormap_array)
    

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
    sample_criteria = getattr(config, 'sample_criteria', None)
    projectors = getattr(config, 'projectors', None)
    image_folder = getattr(config, 'image_folder', None)
    all_images = getattr(config, 'all_images', None)
    preprocess = getattr(config, 'preprocess', None)
    spatial_average = getattr(config, 'spatial_average', None)
    num_workers = getattr(config, 'num_workers', None)
    dot_size_lb = getattr(config, 'dot_size_lb', None)
    dot_size_ub = getattr(config, 'dot_size_ub', None)
    color_std_cutoffs = getattr(config, 'color_std_cutoffs', None)
    map_alpha = getattr(config, 'map_alpha', None)
    grid_size = getattr(config, 'grid_size', None)
    icon_size = getattr(config, 'icon_size', None)
    min_icon_density = getattr(config, 'min_icon_density', None)

    if not isinstance(feature,int):
        feature = feature.to(DEVICE)

    _ = model.to(DEVICE)
    batch_size = dataloader.batch_size
    total_images = len(dataloader.dataset)

    #previously generated data
    norms_data = torch.load('../atlases/%s/data/attribution_norms.pt'%out_name)
    attr_crop_bounds = norms_data['attr_crop_bounds']
    layer_shapes = norms_data['layer_shapes']


    out_data = {}

    for attr_layer in attribution_layers:
        print(attr_layer)
        out_data[attr_layer] = {}

        attr_crop_shape = [attr_crop_bounds[attr_layer][0][1]-attr_crop_bounds[attr_layer][0][0],attr_crop_bounds[attr_layer][1][1]-attr_crop_bounds[attr_layer][1][0]]

        pos_sums = norms_data['pos_l1_attr'][attr_layer].sum(dim=(1,2))
        neg_sums = norms_data['neg_l1_attr'][attr_layer].sum(dim=(1,2))
        target_activations = norms_data['target_activations']

        print('getting data sample')
        sample_conditions = []
        for d_name in sample_criteria:
            if d_name == 'target_activations':
                sample_conditions.append(create_sample_condition(norms_data[d_name], sample_criteria[d_name]))

            elif d_name == 'l1_attr':
                l1_sums = pos_sums+neg_sums
                sample_conditions.append(create_sample_condition(l1_sums, sample_criteria[d_name]))

            else:
                sample_conditions.append(create_sample_condition(norms_data[d_name][attr_layer].sum(dim=(1,2)), sample_criteria[d_name]))
        sample_conditions = torch.stack(sample_conditions)
        combined_conditions = torch.any(sample_conditions, dim=0).numpy()
        sel_indices = np.where(combined_conditions)[0]


        #PLOTTING SELECTION
        print('plotting selection')

        buffer_scale = 0.05  # 10% buffer
        x_range = max(pos_sums) - min(pos_sums)
        y_range = max(neg_sums) - min(neg_sums)
        xlim = [min(pos_sums) - buffer_scale * x_range, max(pos_sums) + buffer_scale * x_range]
        xlim = [0,max(pos_sums) + buffer_scale * x_range]
        ylim = [min(neg_sums) - buffer_scale * y_range, max(neg_sums) + buffer_scale * y_range]
        #ylim = xlim
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
    
        # Selection contour
        x, y = pos_sums[sel_indices].numpy(), neg_sums[sel_indices].numpy()
        # Create a buffer to extend the range
        buffer = .1  # This is a percentage of the total range; adjust as necessary

        # Calculate the range and apply the buffer
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        xmin, xmax = x.min() - (buffer * x_range), x.max() + (buffer * x_range)
        ymin, ymax = y.min() - (buffer * y_range), y.max() + (buffer * y_range)

        # Create a grid of points for KDE
        X_dense, Y_dense = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X_dense.ravel(), Y_dense.ravel()])

        # Perform KDE
        kernel = gaussian_kde(np.vstack([x, y]))
        Z_dense = np.reshape(kernel(positions).T, X_dense.shape)

        # Determine density levels
        min_density = np.min(Z_dense)
        max_density = np.max(Z_dense)
        num_density_lines = 10
        # Adjust the starting point as needed to ensure the outermost contour encompasses all points
        levels = np.linspace(min_density + (max_density - min_density) * 0.1, max_density, num_density_lines)

        # Plot contour lines
        plt.contour(X_dense, Y_dense, Z_dense, levels=levels, colors='black', linewidths=0.5)
        
        # Calculate the Convex Hull for the points
        points = np.vstack([x, y]).T  # Stack your points in the correct shape
        hull = ConvexHull(points)

        # Extract the hull points
        hull_points = points[hull.vertices]

        # Close the loop
        hull_points = np.append(hull_points, [hull_points[0]], axis=0)

        # Fit a spline to the hull points
        tck, u = splprep(hull_points.T, u=None, s=0.0, per=1)

        # Evaluate the spline fits for 1000 equally spaced distance values
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)

        # Plot the smoothed line
        plt.plot(x_new, y_new, linestyle=(0, (5, 10)), linewidth=0.5, color='black') 

        # Scatter plot for all points
        plt.scatter(x=pos_sums, y=neg_sums, c=target_activations, cmap='RdBu_r', s=2)
    #     norm = Normalize(vmin=-max(abs(pos_sums.min()), abs(neg_sums.min()), abs(pos_sums.max()), abs(neg_sums.max())), vmax=max(abs(pos_sums.min()), abs(neg_sums.min()), abs(pos_sums.max()), abs(neg_sums.max())))
    #     plt.scatter(x=pos_sums, y=neg_sums, c=target_activations, cmap='RdBu_r', norm=norm, s=3)
    
        #Select indices
        for idx in sel_indices:
            plt.scatter(pos_sums[idx], neg_sums[idx], s=30, facecolors='none', 
                        edgecolors=(.5, .5, .5, 0.3),
                        linewidths=.5)


        plt.savefig('../atlases/%s/figures/%s_data_selection_plot.png'%(out_name,attr_layer), bbox_inches='tight', dpi=300)
        plt.close()
        plt.clf()


        ###GET SELECTION ATTRIBUTIONS
        print('getting selection attributions')
        select_dataset = SelectImageDataset(image_folder, all_images, sel_indices, transform=preprocess)
        select_dataloader = DataLoader(select_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False, 
                                       num_workers=num_workers) 

        actgrads = torch.zeros(len(sel_indices),layer_shapes[attr_layer][0],attr_crop_shape[0],attr_crop_shape[1])
        #get attribution vector
        with layer_saver(model, target_layer, detach=False) as target_saver:
            with actgrad_extractor(model, [attr_layer], concat=False) as score_saver:
                #for i,img_batch in enumerate(load_select_indices_in_batches(sel_indices, batch_size, image_folder, all_images, preprocess, device)):
                for i, img_batch in enumerate(select_dataloader):

                    if i%10 == 0:
                        print(str(i))

                    model.requires_grad_(True)
                    model.zero_grad()
                    img_batch = img_batch.to(DEVICE)


                    batch_layer_activations = target_saver(img_batch)[target_layer]
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
                        batch_feature_act = torch.matmul(batch_target_activations, torch.tensor(feature).unsqueeze(1)).squeeze()

                    loss = torch.sum(batch_feature_act)
                    loss.backward()

                    activations = score_saver.activations[attr_layer].detach().cpu()
                    gradients = score_saver.gradients[attr_layer].detach().cpu()

                    actgrads_curr = activations*gradients
                    act_grads_curr = actgrads_curr[:,:,attr_crop_bounds[attr_layer][0][0]:attr_crop_bounds[attr_layer][0][1],attr_crop_bounds[attr_layer][1][0]:attr_crop_bounds[attr_layer][1][1]]
                    actgrads[i*batch_size:(i+1)*batch_size] = act_grads_curr

        #PROJECT DATA
        print('projecting data')        

        attr_map = actgrads.clone()
        if spatial_average: attr_map = attr_map.sum(dim=(-2,-1))
        else: attr_map = attr_map.flatten()
        for projector in projectors:
            attr_map = projector.fit_transform(attr_map)

        E_sizes = norms_data['l2_attr'][attr_layer].sum(dim=(1,2))[sel_indices].numpy()
        lower_threshold = np.percentile(E_sizes, 1)
        upper_threshold = np.percentile(E_sizes, 99)
        E_sizes = np.clip(E_sizes, a_min=lower_threshold, a_max=upper_threshold)
        E_min = np.min(E_sizes)
        E_max = np.max(E_sizes)
        E_sizes = dot_size_lb + ((E_sizes - E_min) * (dot_size_ub - dot_size_lb)) / (E_max - E_min)

        vmin = target_activations[sel_indices].mean()-color_std_cutoffs[0]*target_activations[sel_indices].std()  # or any specific minimum value you wish
        vmax = target_activations[sel_indices].mean()+color_std_cutoffs[1]*target_activations[sel_indices].std()          
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

        layout = normalize_layout(attr_map)


        plot_indices = np.argsort(torch.abs(target_activations[sel_indices]).numpy()) # Get indices sorted by z
        sorted_x = layout[:,0][plot_indices]
        sorted_y = layout[:,1][plot_indices]
        sorted_acts = target_activations[sel_indices][plot_indices] # Not used for plotting, but here for completeness
        sorted_E_sizes = E_sizes[plot_indices]

        plt.figure(figsize=(10, 10))  # Set both dimensions to the same value

        plt.scatter(sorted_x,sorted_y,
                c=sorted_acts, 
                cmap=modified_rdbu, 
                norm=norm,
                alpha= map_alpha,
                s=sorted_E_sizes
                )
        plt.savefig('../atlases/%s/figures/%s_attribution_map.png'%(out_name,attr_layer), bbox_inches='tight', dpi=300)
        plt.close()
        plt.clf()

        print('getting icon attributions')
        xs = layout[:, 0]
        ys = layout[:, 1]

        grid_layout = grid(xpts=xs, ypts=ys, 
                       grid_size=grid_size,  
                       x_extent=(0., 1.0), y_extent=(0., 1.0)
                      )

        icons = []
        images_per_icon = []
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                indices = grid_layout[x, y]
                if len(indices) > min_icon_density:
                    average_actgrad = np.average(actgrads[indices], axis=0)
                    icons.append((average_actgrad, x, y))
                    images_per_icon.append(len(indices))

        icons = np.asarray(icons,dtype=object)

        save_obj = {
                    'sel_indices':sel_indices,
                    'attr_map':attr_map,
                    'layout':layout,
                    'icons':icons,
                    'grid_layout':grid_layout
                    }
        torch.save(save_obj,'../atlases/%s/data/%s_atlas.pt'%(out_name,attr_layer))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <cfg_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    main(config_path)