import argparse
import subprocess
import os
import importlib.util

# Setup argument parser to accept configuration path from command line
parser = argparse.ArgumentParser(description='Generate atlas from configuration.')
parser.add_argument('--config', type=str, help='Path to configuration file', required=True)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing data if out_name is the same')
parser.add_argument('--skip-energies', action='store_true', help='skip first step of getting attribution energies')
parser.add_argument('--skip-projection', action='store_true', help='skip second step of umap projection')
parser.add_argument('--skip-featureviz', action='store_true', help='skip third step of visualizing icons')
parser.add_argument('--skip-render', action='store_true', help='skip final step of compiling the atlas')
args = parser.parse_args()

cfg_path = args.config
overwrite = args.overwrite
skip_energies = args.skip_energies
skip_projection = args.skip_projection
skip_featureviz = args.skip_featureviz
skip_render = args.skip_render

# Load the configuration module from the specified path
cfg_path = os.path.abspath(cfg_path)
spec = importlib.util.spec_from_file_location("config_module", cfg_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
out_name = getattr(config, 'out_name', None)

if not overwrite:
    assert not os.path.exists('./atlases/'+out_name), f'Atlas {out_name} already exists!'

print(f'making folder ./atlases/{out_name}')
os.makedirs(f'./atlases/{out_name}', exist_ok=True)
os.makedirs(f'./atlases/{out_name}/figures/', exist_ok=True)
os.makedirs(f'./atlases/{out_name}/data/', exist_ok=True)
subprocess.call(f'cp {cfg_path} ./atlases/{out_name}', shell=True)
if os.path.exists('logs/{out_name}.out'):
    subprocess.call(f'rm logs/{out_name}.out', shell=True)

with open(f'logs/{out_name}.out', 'a') as log_file:
    if not skip_energies:
        print('GENERATING ATTRIBUTION ENERGIES')
        subprocess.run(['python', 'generation_scripts/gen_attribution_energies.py', cfg_path],
                    stdout=log_file,  
                    stderr=log_file
                    )
    else:
        print('skipping attribution energy')  
    if not skip_projection:
        print('GENERATING PROJECTION') 
        subprocess.run(['python', 'generation_scripts/gen_projection.py', cfg_path],
                    stdout=log_file,  
                    stderr=log_file
                    ) 
    else:
        print('skipping projection') 
    if not skip_featureviz:
        print('GENERATING FEATURE VISUALIZATIONS')
        subprocess.run(['python', 'generation_scripts/submit_batch_visualizations.py', cfg_path],
                    stdout=log_file,  
                    stderr=log_file
                    ) 
    else:
        print('skipping feature viz') 

    if not skip_render:     
        print('GENERATING FINAL ATLAS')
        subprocess.run(['python', 'generation_scripts/atlas_render.py', cfg_path],
                    stdout=log_file,  
                    stderr=log_file
                    )  
    else:
        print('skipping final atlas')        
