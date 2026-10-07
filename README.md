# Degradation-Aware Unfolding Network with Sparsity-Gated Experts

DUMoE provides training and evaluation for image compressive sensing (ICS),
CS-MRI, and simulated and real hyperspectral snapshot compressive imaging (SCI).
Run all commands below from the repository root.

## Installation

Use Python 3.10 or later and install a matching PyTorch/torchvision pair for your
device, then install the dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install -r requirements-lpips.txt      # ICS LPIPS metric
pip install -r requirements-training.txt   # EMA and TensorBoard
```

Evaluation selects CUDA when available; use `--device cpu` or `--device cuda:N`
to choose a device. Training defaults to CUDA.

## Pretrained models

Extract `DUMoE-pretrained.zip` into the repository root. The checkpoints are
placed under `ics/model/`, `csmri/model/`, `sci/sim/model/`, and `sci/real/model/`.
ICS sampling matrices are included. Prepare datasets and MRI/SCI masks separately
using the formats below.

## Image compressive sensing (ICS)

### Data

Create `ics/data/test/{dataset}/` and place test images there, for example
`ics/data/test/Set14/`. JPEG, PNG, TIFF, and BMP images are supported. Prepare
separate image folders for training and validation.

Supported sampling ratios are `1`, `4`, `10`, `25`, `30`, `40`, and `50` percent.
The corresponding matrices are in `ics/data/sampling_matrix/`.

### Evaluation

```bash
python ics/main_test.py --dataset Set14 --ratio 25
```

Results are saved to `ics/results/dumoe/{dataset}/{ratio}/`, including
reconstructed images and `results.csv` with luminance PSNR/SSIM. Add `--lpips`
to report LPIPS. Use `--data_dir` for another test directory and `--checkpoint`
to select a pretrained or trained checkpoint.

### Training

```bash
python ics/main_train.py --data_path /path/to/train --eval_data_path /path/to/val --cs_ratio 25 --output_dir runs/ics
```

Training converts images to grayscale and uses random crops, rotations, and
flips. The default crop size is 96; `--input_size` must be a multiple of 32.
Validation images are resized to that crop size. Checkpoints are written to
the selected output directory, including `checkpoint-dumoe-{ratio}-best.pth`.

## CS-MRI

### Data

Create `csmri/data/test/Brain_test/` for test images. Inputs are 8-bit grayscale
PNGs or MAT files containing a real-valued 2D `data` array in `[0, 1]`.
Evaluation applies a 256 x 256 center crop; images must cover that size.
Training and validation folders use the same image formats.

Place masks at `csmri/data/mask/256/Radial_{ratio}.mat`. Each file contains a
256 x 256 `mask_matrix` in the unshifted FFT convention. Supported evaluation
ratios are `5`, `10`, `20`, `30`, and `40` percent.

### Evaluation

```bash
python csmri/main_test.py --dataset Brain_test --ratio 10
```

Results are saved to `csmri/results/dumoe/{dataset}/{ratio}/`, including
reconstructed images and `results.csv` with PSNR/SSIM and inference time.
Use `--checkpoint`, `--data_dir`, or `--mask_path` to override the default inputs.
The separate brain/radial models are available under `csmri/model/brain_radial/`.

### Training

```bash
python csmri/main_train.py --data_path /path/to/train --eval_data_path /path/to/val --input_size 256 --mask_type Radial --cs_ratio 10 --output_dir runs/csmri
```

Training uses a center crop and horizontal/vertical flips. The supplied default
crop is 320; the example selects 256 to match the evaluation layout. Crops must
be divisible by four and covered by every input image.

The mask must match the crop size. Its default location is
`csmri/data/mask/{input_size}/{mask_type}_{ratio}.mat`, or specify `--mask_path`.
Radial masks contain `mask_matrix`; Cartesian masks contain `mask`. Training
checkpoints include `checkpoint-dumoe-{ratio}-best.pth` in the output directory.

## Simulated SCI

### Data

Create this layout under `sci/sim/data/SCI/`, or provide it through `--data_root`:

```text
SCI/
  TSA_simu_data/
    mask_3d_shift.mat
    Truth/
      scene01.mat
      ...
  cave_1024_28/
    scene001.mat
    ...
```

Truth files contain `img` with shape 256 x 256 x 28 and values in `[0, 1]`.
The mask contains `mask_3d_shift` with shape 256 x 310 x 28. CAVE cubes are
needed for training and contain 16-bit-scale `img` or `img_expand` arrays.

### Evaluation

```bash
python sci/sim/main_test.py --data_root /path/to/SCI --outf results/sci-sim
```

The default model is `sci/sim/model/dumoe_sci_sim.pth`. The output directory
contains `dumoe.mat` with `pred` and `truth` (N x 256 x 256 x 28), and
`results.csv` with PSNR/SSIM. Use `--pretrained_model_path` to select another model.

### Training

```bash
python sci/sim/train.py --data_root /path/to/SCI --outf runs/sci-sim --checkpoint_gradients
```

Training uses CAVE crops and the truth scenes for validation. It saves
`latest.pth` and the best validation model as `best.pth`. The optional
`--checkpoint_gradients` flag reduces activation memory by recomputing stages
during backward.

## Real SCI

### Data

Create this layout under `sci/real/data/SCI/`, or provide it through `--data_root`:

```text
SCI/
  TSA_real_data/
    mask_3d_shift.mat
    Measurements/
      scene1.mat
      scene2.mat
      ...
  cave_1024_28/
  KAIST_CVPR2021/
```

Measurement files contain `meas_real` with shape 660 x 714. The mask contains
`mask_3d_shift` with shape 660 x 714 x 28. Training requires CAVE files with
16-bit-scale `img` or `img_expand` arrays and KAIST files with `HSI` in `[0, 1]`.

### Evaluation

```bash
python sci/real/main_test.py --data_root /path/to/SCI --outf results/sci-real
```

The default model is `sci/real/model/dumoe_sci_real.pth`. Outputs include
per-scene MAT files containing `res` and `Real_result.mat` containing `pred`.
The default reconstruction is 660 x 660 x 28. Custom `--height` and `--width`
must be equal and divisible by four. Use `--pretrained_model_path` to select
another model.

### Training

```bash
python sci/real/train.py --data_root /path/to/SCI --outf runs/sci-real --pretrained_model_path sci/sim/model/dumoe_sci_initial.pth
```

Real SCI training requires initialization weights and saves `latest.pth` in
the output directory. Add `--checkpoint_gradients` to reduce activation memory.

## Common training options

For ICS/CS-MRI, use `--epochs`, `--batch_size`, and `--lr` to adjust training.
`--init_checkpoint` initializes model weights; `--resume` restores a compatible
training checkpoint. Use `--disable_eval true` when validation data is unavailable.

For SCI, use `--max_epoch`, `--batch_size`, and `--learning_rate`.
The effective batch is processed one sample at a time for sparse routing.
Use a new output directory for fresh training. Run any entry point with `--help`
to see its complete options.
