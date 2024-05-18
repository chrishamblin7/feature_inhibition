### Attribution Atlas

This folder has code for generating attribution atlases. Here's how:

1st: make a config file

 `cp configs/config_template my_config.py`
 edit the config file to your use case


 2nd: generate the atlas

 `python gen_atlas.py --config my_config.py`

Intermediate data and your desired figures will appear in the 'atlases' folder
