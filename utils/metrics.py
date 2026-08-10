import torch
import numpy as np
from sklearn.metrics import jaccard_score, f1_score
import torch.nn.functional as F

def compute_segmentation_metrics(pred_masks, true_masks, threshold=0.5):
    """
    محاسبه متریک‌های ارزیابی برای segmentation
    
    Args:
        pred_masks: پیش‌بینی‌های مدل [frames, H, W] یا [frames, 1, H, W]
        true_masks: ماسک‌های واقعی [frames, H, W]
        threshold: آستانه برای باینری کردن پیش‌بینی‌ها
    """
    # تبدیل به numpy
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(true_masks, torch.Tensor):
        true_masks = true_masks.cpu().numpy()
    
    # اگر pred_masks دارای بعد اضافی است (مثلاً [frames, 1, H, W])
    if len(pred_masks.shape) == 4:
        pred_masks = pred_masks.squeeze(1)  # [frames, H, W]
    
    # اطمینان از شکل‌های یکسان
    if pred_masks.shape != true_masks.shape:
        # اگر شکل‌ها متفاوت است، reshape کن
        if len(pred_masks.shape) == 3 and len(true_masks.shape) == 3:
            # هر دو 3 بعدی هستند
            if pred_masks.shape[1:] != true_masks.shape[1:]:
                # resize pred_masks به اندازه true_masks
                from scipy.ndimage import zoom
                h_ratio = true_masks.shape[1] / pred_masks.shape[1]
                w_ratio = true_masks.shape[2] / pred_masks.shape[2]
                pred_masks = zoom(pred_masks, (1, h_ratio, w_ratio), order=1)
    
    # باینری کردن
    pred_binary = (pred_masks > threshold).astype(np.float32)
    true_binary = (true_masks > threshold).astype(np.float32)
    
    # محاسبه متریک‌ها
    num_frames = pred_binary.shape[0]
    
    iou_scores = []
    dice_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    
    for t in range(num_frames):
        pred_flat = pred_binary[t].flatten()
        true_flat = true_binary[t].flatten()
        
        # اطمینان از طول یکسان
        min_len = min(len(pred_flat), len(true_flat))
        pred_flat = pred_flat[:min_len]
        true_flat = true_flat[:min_len]
        
        # IoU
        intersection = np.sum(pred_flat * true_flat)
        union = np.sum(pred_flat) + np.sum(true_flat) - intersection
        iou = intersection / (union + 1e-7)
        iou_scores.append(iou)
        
        # Dice
        dice = (2 * intersection) / (np.sum(pred_flat) + np.sum(true_flat) + 1e-7)
        dice_scores.append(dice)
        
        # Accuracy
        acc = np.mean(pred_flat == true_flat)
        accuracy_scores.append(acc)
        
        # Precision و Recall
        tp = intersection
        fp = np.sum(pred_flat) - tp
        fn = np.sum(true_flat) - tp
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    return {
        'iou': np.mean(iou_scores),
        'dice': np.mean(dice_scores),
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores)
    }

def compute_frame_accuracy(pred_masks, true_masks, threshold=0.5):
    """محاسبه دقت در سطح فریم"""
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(true_masks, torch.Tensor):
        true_masks = true_masks.cpu().numpy()
    
    pred_binary = (pred_masks > threshold).astype(np.float32)
    true_binary = (true_masks > threshold).astype(np.float32)
    
    correct = (pred_binary == true_binary).float().mean()
    return correct.item()
