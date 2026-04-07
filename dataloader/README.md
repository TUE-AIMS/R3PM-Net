# Dataloaders

This directory contains scripts to generate Simulated datasets for training and testing.
It uses functionalities from `tools` folder to:

`generate_dataset`

- load and downsample data
- compute normals if needed
- apply random transformations
- add augmentations (noise, outliers and occlusion)
- save point clouds

**Note: Two input point clouds must have same length.** 

`generate_dataset_dict`

- save generated dataset in dictionaries suitable to train models (following Learning3d requirments)
- checks dimensions (to meet Learning3d requirments)

`combine_dataset_dict`

- shuffle and combine all generated dictionaries into one 
- split train and test sets

## How to generate datasets?

Modify the `args.txt` file to contain the correct paths and other specifications e.g. downsampling rate, noise level, etc. Other default arguments in `data_dict_generator.py` can also be changed.

1. Generate transformed target point clouds + GT transforms
Change `--action in dataloader/args.txt` to `generate_dataset` and run:
```
python dataloader/data_dict_generator.py @dataloader/args.txt
```

2. Generate train/test .pkl dicts 
Change `--action in dataloader/args.txt` to `generate_dataset_dict`, then run the above script again.

3. Combine multiple object dicts 
Set `--action combine_dataset_dict` and run again to get train and test `dict.pkl` files.

### Manual Run (without args.txt)
Optinally you can manually run: 
```
python dataloader/data_dict_generator.py \
  --pcdPath /path/to/source_scan.pcd \
  --cadPath /path/to/object.stl \
  --name teeth \
  --action generate_dataset \
  --every_k_points 100 \
  --num_transformation 50 \
  --angles 0 90 180 \
  --translation_range -1 1 \
  --index 0 \
  --noise_level 0 \
  --outlier_level 0 \
  --outlier_bounds -0.05 0.05 \
  --occ_level 0 \
  --save_path data/simulators
```
then:
```
python dataloader/data_dict_generator.py \
  --pcdPath /path/to/source_scan.pcd \
  --cadPath /path/to/object.stl \
  --name teeth \
  --action generate_dataset_dict \
  --dataset_size 50 \
  --index 0 \
  --save_path data/simulators
```