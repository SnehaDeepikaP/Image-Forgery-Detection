# Proposed Solution: Hybrid CNN + Vision Transformer Dual-Task Model

This implementation plan outlines the steps to integrate the proposed hybrid architecture for joint image manipulation classification and localization in the notebook [`Final Year.ipynb`](file:///f:/%23sem7/22AIE498%20Project%20Phase%201/Project/CNN-vs-ViT-for-Image-Manipulation-Detection-main/Final%20Year.ipynb). The proposed solution extends the base paper implementation by combining a CNN (ResNet-50) and a Vision Transformer (Swin-T) into a unified backbone with multi-scale feature fusion and a segmentation decoder, trained under a curriculum learning scheme with a composite loss function.

## User Review Required

> [!IMPORTANT]
> - **Virtual Environment**: We will utilize the user-configured virtual environment `gpu_env` located at `F:\#sem7\22AIE498 Project Phase 1\Project\gpu_env` for executing code and installing any needed helpers.
> - **Non-Destructive Changes**: All existing cells in `Final Year.ipynb` will remain untouched. We will append the new implementation cells to the end of the notebook.
> - **Computation Time**: The hybrid model contains both ResNet-50 and Swin-T. Training will take longer per epoch compared to the individual models. We will configure default training epochs (e.g., 10 epochs) and batch size (e.g., 16/32) with a safety early stopping mechanism to optimize time.

## Open Questions

> [!NOTE]
> None. We will implement standard defaults for loss coefficients ($\alpha=1.0, \beta=1.0, \gamma=1.0, \delta=0.5$), which can be tuned as hyperparameters.

## Proposed Changes

### Jupyter Notebook Integration

#### [MODIFY] [Final Year.ipynb](file:///f:/%23sem7/22AIE498%20Project%20Phase%201/Project/CNN-vs-ViT-for-Image-Manipulation-Detection-main/Final%20Year.ipynb)
We will add three new cells at the end of the notebook under a new header `# EXP 9: PROPOSED HYBRID CNN-ViT DUAL-TASK MODEL`.

1. **Cell 15 (Markdown)**: Documentation of the proposed solution architecture and loss equations.
2. **Cell 16 (Code - Dataset & Architecture)**:
   - **`HybridManipulationDataset`**: A PyTorch Dataset loading images and masks. For pristine images, it returns an all-zero mask. For manipulated images, it loads the ground-truth mask using the direct path matching logic from `EXP 6_7`.
   - **`HybridCNNViTClassifier`**:
     - **Backbone**: Parallel ResNet-50 and Swin-T feature extractors.
     - **Multi-scale Feature Fusion**: Concatenates intermediate features at stages corresponding to resolutions $56\times56, 28\times28, 14\times14, 7\times7$.
     - **Segmentation Decoder**: A U-Net/FPN-style decoder with lateral skip connections and upsampling layers to produce a $224\times224$ single-channel segmentation mask.
     - **Classification Head**: Aggregates bottleneck features using Global Average Pooling (GAP) and classifies via a linear projection head.
   - **Composite Loss**:
     - Binary Cross-Entropy with Logits (Classification).
     - Dice Loss + IoU Loss + Boundary Loss (Segmentation).
     - **Boundary Loss**: Implemented using MaxPool2d-based morphological boundary extraction.
     - **Attention-guided Refinement**: Computes MSE between internal spatial attention maps (channel-averaged features) of the encoder stages and the ground-truth mask as auxiliary supervision.
3. **Cell 17 (Code - Curriculum Training & Evaluation)**:
   - **Curriculum Training**: Starts training with lower weights on the segmentation loss terms (e.g., weight 0.1) for the first few epochs so the network establishes stable classification features, then scales the segmentation loss weights to 1.0.
   - **Evaluation**: Calculates Test Set classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and localization metrics (IoU, Dice, Boundary Distance).
   - Plots prediction heatmaps alongside ground-truth masks for visual verification.

---

## Verification Plan

### Automated Tests
We will execute the newly added notebook code inside the `gpu_env` environment using a python script to verify:
1. **Model Instantiation**: Instantiate the `HybridCNNViTClassifier` and verify feature shapes.
2. **Loss Function Correctness**: Run mock predictions through `DiceLoss`, `IoULoss`, and `BoundaryLoss` to ensure value ranges are correct.
3. **Smoke Test Training**: Run a test loop on a small subset (e.g., 5 batches) to verify forward and backward passes.
