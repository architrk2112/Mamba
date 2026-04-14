from __future__ import annotations

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute Normalized Mean Squared Error (NMSE).
    
    Args:
        pred: Prediction tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        eps: Small epsilon for numerical stability
    
    Returns:
        NMSE value as float
    """
    num = torch.mean((pred - target) ** 2)
    den = torch.mean(target ** 2) + eps
    return float((num / den).item())


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    """
    Compute Peak Signal to Noise Ratio metric (PSNR) using torchmetrics.
    
    Args:
        pred: Prediction tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        data_range: The data range of the input image (default: 1.0)
    
    Returns:
        PSNR value in dB as float
    """
    if data_range is None:
        data_range = 1.0
    
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(pred.device)
    psnr_value = psnr_metric(pred, target)
    return float(psnr_value.item())


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    """
    Compute Structural Similarity Index Metric (SSIM) using torchmetrics.
    
    Args:
        pred: Prediction tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        data_range: The data range of the input image (default: 1.0)
    
    Returns:
        SSIM value as float
    """
    if data_range is None:
        data_range = 1.0
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(pred.device)
    ssim_value = ssim_metric(pred, target)
    return float(ssim_value.item())

