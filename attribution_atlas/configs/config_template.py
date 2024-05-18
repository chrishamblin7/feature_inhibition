#config file for specifying parameters for generating the atlas

out_name = 'OUT_NAME_PLACEHOLDER'

DEVICE = 'cuda:0'

#model
from faccent.modelzoo import inceptionv1_decomposed
model = inceptionv1_decomposed(pretrained=True)
MODEL_INPUT_SIZE = model.model_input_size
MODEL_INPUT_RANGE = model.model_input_range

#feature to target
target_layer = 'TARGET_LAYER_PLACEHOLDER'    #from circuit_explorer.utils import convert_relu_layers
attribution_layers = ATTRIBUTION_LAYERS_PLACEHOLDER       #attribution layer placeholder (should be list)
feature =  FEATURE_PLACEHOLDER     #integer or vector (of target_layer dimension)
position = 'middle'   #position in target layer to generate data from

#dataset (large data set from which we will sample based on some criteria)
image_folder = '/mnt/data/datasets/imagenet/train100/'
batch_size = 40

if '/val' in image_folder:
    class_folders = True
else:
    class_folders = False
 
import os

#preprocess transformations
from torchvision.transforms import Compose,Resize,ToTensor
from torch.utils.data import DataLoader
from faccent.utils import LargestCenterCrop, image_data
from faccent.transform import range_normalize

#from PIL preprocess
transforms = []
transforms.append(LargestCenterCrop())
transforms.append(Resize(MODEL_INPUT_SIZE))
transforms.append(ToTensor())
transforms.append(range_normalize(MODEL_INPUT_RANGE))
preprocess = Compose(transforms)

#from tensor preprocess
from faccent import transform
simple_preprocess = transform.compose([
                                        transform.resize(MODEL_INPUT_SIZE),
                                        transform.range_normalize(MODEL_INPUT_RANGE)   
                                        ])


num_workers = 4

dset = image_data(image_folder, 
                transform=preprocess,
                class_folders=class_folders)
dataloader = DataLoader(dset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers= num_workers,
                    pin_memory=True,
                    sampler= None
                    )   

all_images = dset.img_names
all_images.sort()
# all_images = os.listdir(image_folder)
# all_images.sort()  


#Sample Criteria
'''
well sample from the atlas based on some criteria.
Given the sorted list of activations and attribution-norms,
dictionary values specify the indices we want to select.
For example:

 {'target_activations':[list(range(-1,-5000,-1)),
                        list(range(5000))],
  'attribution_L2':[list(range(10000))]}

would select the images with top-5000 activations, the bottom-5000 activations,
and the images with the top-10000 L2 norm of the attribution vector.

options are target_activations, l2_attr, l1_attr,
pos_l2_attr, neg_l2_attr,pos_l1_attr, neg_l1_attr.
pos/neg will check the norm of only those positive and negative terms of the attribution vector.
'''  

# sample_criteria = {'target_activations':
#                         [
#                          list(range(-1,-2000,-1)),
#                          list(range(2000))
#                          ],
#                     'l2_attr':
#                         [
#                          list(range(-1,-10000,-1))
#                          ]
#                    }

sample_criteria = {
                    'l2_attr':
                        [
                         list(range(-1,-10000,-1))
                         ]
                   }



#projector parameters
from umap import UMAP

#projectors will be called in series on the data,
#so could have a pca then umap in the list for example
projectors = [UMAP(n_neighbors=10, min_dist=0.1, n_components=2,metric="cosine")]
spatial_average = True #average over spatial dimensions of attribution before umap
dot_size_lb = .05
dot_size_ub = 4
color_std_cutoffs = [7,7]
map_alpha = .6

#atlas hyperparams
grid_size = (30,30)
icon_size= 160
min_icon_density=5
act_color_cutoff = [7,7]   #standard deviations of activations at which to color pure blue/red
crop_adjustment = 0    #how much to shift the crop bounds by (+ for out, - for in)
dpi=600
center_activations = True

import torch
#feature viz parameters
canvas_size = (512,512)
out_thresholds = [201]
copy_batch = False
seed=None
featureviz_batch_size = 30
optimizer = lambda params: torch.optim.Adam(params, lr=.06)
cosine_power = 2.
negative_boost = 1.
trans_p = 10









