# LoRT Tensor Completion Demo

This repository provides MATLAB demo scripts for evaluating the **Low-Rank Tensor Transitions (LoRT)** framework under various experimental settings for visual tensor completion tasks. The experiments focus on analyzing the impact of the number of source tasks and sampling rates.

> ⚠️ **Ongoing Improvements Planned**  
> As the camera-ready timeline was constrained, several aspects are currently under development and will be added progressively, including:
> - Broader evaluation metrics (e.g., SSIM, LPIPS)  
> - Extended algorithmic benchmarks  
> - Support for distributed node computation  
> - Deeper robustness and ablation studies


## Folder Structure

```
.
├── data/                         % Folder to store input video tensors and masks
├── libs/                         % Dependencies and utility functions (e.g., TNN, ADMM solvers)
├── Demo_test_on_different_source_tasks.m
├── Demo_test_on_different_SR_for_target_tasks.m
├── Demo_test_on_different_SR_Source_tasks.m
```

## Demo Scripts

- **Demo_test_on_different_source_tasks.m**  
  Varies the number of source tasks $K$ to evaluate transferability and performance trends of LoRT in spatiotemporal tensor completion.

- **Demo_test_on_different_SR_for_target_tasks.m**  
  Fixes the source tensors and changes the sampling rate (SR) of the **target** task to study LoRT’s robustness under extreme sparsity.

- **Demo_test_on_different_SR_Source_tasks.m**  
  Fixes the target task and changes the SR of the **source** tasks to analyze the sensitivity of LoRT to source data availability.

## Dependencies

Ensure the following are available or included in `libs/`:
- Tensor completion solvers (e.g., t-SVD, TNN)
- Proximal operators and ADMM-based optimizers
- Preprocessing utilities for loading and masking video tensors

## Notes

- All video data should be preprocessed into 3D tensors and stored in the `data/` directory.
- Each script saves its results in `.mat` or `.png` format.

## Citation

If you use this code, please cite our paper on LoRT:

> Wang et al. *Low-Rank Tensor Transitions for Transferable Tensor Regression*. ICML 2025.
