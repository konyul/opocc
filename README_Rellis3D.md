# OpenOccupancy for Rellis-3D Dataset

This is a modified version of OpenOccupancy to work with the Rellis-3D dataset, configured to output 3 classes similar to the GaussianFormer implementation.

## Setup

1. Ensure you have the Rellis-3D dataset in OCCFusion format at:
   ```
   /mnt/sdb/kypark/OCCFusion/data/Rellis-3D/
   ```

2. Ensure the data info files are at:
   ```
   /mnt/sdb/kypark/OCCFusion/data_infos/occfusion_train_add_geom.pkl
   /mnt/sdb/kypark/OCCFusion/data_infos/occfusion_val_final_0_2000.pkl
   ```

## Configuration

The main configuration file is located at:
```
projects/configs/baselines/Multimodal-R50_img1600_rellis3d.py
```

### Key Settings:
- **Classes**: 3 classes (empty, traversable, non-traversable)
- **Input Size**: 1600x864 (matching GaussianFormer)
- **Point Cloud Range**: [-25.6, -12.8, -1.6, 0.0, 12.8, 1.6]
- **Occupancy Grid Size**: 256x256x32
- **Camera**: Single front camera (Rellis-3D setup)

## Training

To train the model on Rellis-3D:
```bash
./train_rellis3d.sh
```

The training results will be saved in:
```
work_dirs/rellis3d_multimodal_3class/
```

## Evaluation

To evaluate a trained model:
```bash
./eval_rellis3d.sh [checkpoint_path]
```

Example:
```bash
./eval_rellis3d.sh work_dirs/rellis3d_multimodal_3class/latest.pth
```

## Model Architecture

### Key Components:
1. **Image Backbone**: ResNet-50 with FPN
2. **LiDAR Encoder**: Sparse 3D convolution encoder
3. **Fusion Module**: ConvFuser for multi-modal fusion
4. **Occupancy Head**: Modified OccHeadRellis for 3-class prediction

### Loss Configuration:
- Cross-entropy loss with class weights: [0.01, 5.0, 1.0]
- Optional Lovasz loss (disabled by default)
- Semantic and geometric scale losses

## Class Weights

The model uses weighted cross-entropy loss to handle class imbalance:
- Empty: 0.01 (very common)
- Traversable: 5.0 (important for navigation)
- Non-traversable: 1.0 (baseline)

## Notes

1. The model is configured for single-GPU training by default. Modify `GPUS` in the scripts for multi-GPU training.

2. The data loader expects Rellis-3D data in OCCFusion format with:
   - Single front camera images
   - 5D LiDAR points (x, y, z, intensity, ring)
   - Occupancy ground truth as `.npy` files

3. The model outputs 3 classes:
   - 0: Empty space
   - 1: Traversable terrain
   - 2: Non-traversable obstacles

4. Training parameters match GaussianFormer:
   - Learning rate: 2e-4
   - Optimizer: AdamW
   - Batch size: 1
   - Epochs: 24

## Troubleshooting

### Import Errors
If you encounter import errors, ensure:
1. You have installed mmdetection3d in the correct environment
2. The PYTHONPATH includes the OpenOccupancy directory
3. All dependencies are installed

### OccHead Registration Error
The code uses a custom `OccHeadRellis` instead of the default `OccHead` to support 3 classes. The original `OccHead` is hardcoded for 17 NuScenes classes.

### Data Loading Issues
Ensure the data paths in the config file match your actual data location:
- `data_root`: Base path to Rellis-3D data
- `train_ann_file`: Path to training annotation pickle file
- `val_ann_file`: Path to validation annotation pickle file