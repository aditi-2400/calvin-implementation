# CALVIN: A Benchmark for Language-Conditioned Policy Learning

This repository contains my reproduction of the **CALVIN benchmark**, following the paper:

> Mees, O., Memmesheimer, R., and Burgard, W.  
> [**CALVIN: A Benchmark for Language-Conditioned Policy Learning**](https://arxiv.org/abs/2112.03227)

The project focuses on training and evaluating **language-conditioned robotic policies** using **unstructured play data** and **language annotations**.


## Project Structure
- `calvin_models/` – Original cloned CALVIN repo codebase
- `dataset/` – Debug dataset used for testing (`calvin_debug_dataset`)
- `calvin_models/runs/` – Training run outputs (events, checkpoints)
- `calvin_models/calvin_agent/training.py` – Entry point for training

## Setup

### 1. Create environment
```bash
conda create -n calvin_venv python=3.8
conda activate calvin_venv
```

### 2. Install dependencies
```bash
sh install.sh
```

### MacOS Fix: xxhash → pyhash

The repo imports `xxhash`, which does not compile well on macOS.  
Replace it with `pyhash` in the code:

```python
try:
    import xxhash
except ImportError:
    import pyhash as xxhash

```

On Apple Silicon (M1/M2/M3), PyTorch uses **MPS backend**.  
Ensure you run training with **float32** precision since MPS does not support float64.

## Training

Example command:
```bash
python training.py \
  datamodule.root_data_dir="dataset/calvin_debug_dataset" \
  datamodule/datasets=vision_lang \
  +datamodule.num_workers=0 \
  +trainer.fast_dev_run=1 \
  +trainer.num_sanity_val_steps=0 \
  logger=tb_logger \
  "~callbacks/rollout" "~callbacks/rollout_lh" \
  trainer.precision=32
```


## TensorBoard
To visualize training:
```bash
tensorboard --logdir ./calvin_models/runs/<DATE>/<TIME>/play_lmp
```
Open [http://localhost:6006](http://localhost:6006) in your browser.

## Results
- **Successfully reproduced CALVIN baseline training** on the debug dataset.
- Validated logging with **TensorBoard**.
- Ensured model trains on **MPS backend** with float32 precision.

## Debugging Journey
Some key issues faced during reproduction:
- **Hydra Config Errors** → fixed by using `+` overrides and proper logger configs.  
- **Logger not found** → used `logger=tb_logger`.  
- **MPS Float64 error** → forced `trainer.precision=32`.  
- **pyhash dependency missing** → installed xxhash.  
- **TensorBoard crash** → resolved by upgrading `protobuf`.  

These fixes are now included in this README for future reproducibility.

## Citation
If you use CALVIN in your research, please cite:
```bibtex
@article{mees2021calvin,
  title={CALVIN: A Benchmark for Language-Conditioned Policy Learning},
  author={Mees, Oier and Memmesheimer, Raphael and Burgard, Wolfram},
  journal={arXiv preprint arXiv:2112.03227},
  year={2021}
}
```

---

## Links
- [Original CALVIN Repo](https://github.com/mees/calvin)  
- [Paper](https://arxiv.org/abs/2112.03227)
