"""
Model evaluation metrikleri.

Author: sulegogh
Date: 2025-11-11
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from loguru import logger


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    dataset_name: str = "Dataset"
) -> Dict:
    """
    Model'i değerlendir ve metrikleri hesapla.
    
    Args:
        y_true: Gerçek labels
        y_pred: Tahmin edilen labels
        y_pred_proba: Tahmin edilen probabilities (optional)
        dataset_name: Dataset ismi (train, val, test)
        
    Returns:
        dict: Metrikler
    """
    logger.info(f"{dataset_name} metrikleri hesaplanıyor...")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_samples': len(y_true),
    }
    
    # ROC AUC (multi-class için one-vs-rest)
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(
                y_true,
                y_pred_proba,
                multi_class='ovr',
                average='weighted'
            )
            metrics['roc_auc'] = roc_auc
        except Exception as e:
            logger.warning(f"ROC AUC hesaplanamadı: {e}")
            metrics['roc_auc'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    logger.info(f"  ✓ {dataset_name}: Acc={accuracy:.4f}, F1={f1:.4f}")
    
    return metrics


def print_metrics(metrics: Dict):
    """
    Metrikleri yazdır.
    
    Args:
        metrics: Metrik dict'i
    """
    logger.info("=" * 60)
    logger.info(f"📊 {metrics['dataset']} METRİKLERİ")
    logger.info("=" * 60)
    logger.info(f"  Samples:   {metrics['n_samples']:,}")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    if metrics.get('roc_auc') is not None:
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    logger.info("=" * 60)


def compare_metrics(train_metrics: Dict, val_metrics: Dict, test_metrics: Optional[Dict] = None):
    """
    Train/val/test metriklerini karşılaştır.
    
    Args:
        train_metrics: Train metrikleri
        val_metrics: Validation metrikleri
        test_metrics: Test metrikleri (optional)
    """
    logger.info("\n" + "=" * 70)
    logger.info("📊 METRİK KARŞILAŞTIRMASI")
    logger.info("=" * 70)
    
    comparison_data = {
        'Dataset': ['Train', 'Val'],
        'Samples': [train_metrics['n_samples'], val_metrics['n_samples']],
        'Accuracy': [train_metrics['accuracy'], val_metrics['accuracy']],
        'Precision': [train_metrics['precision'], val_metrics['precision']],
        'Recall': [train_metrics['recall'], val_metrics['recall']],
        'F1 Score': [train_metrics['f1_score'], val_metrics['f1_score']],
    }
    
    if train_metrics.get('roc_auc') and val_metrics.get('roc_auc'):
        comparison_data['ROC AUC'] = [train_metrics['roc_auc'], val_metrics['roc_auc']]
    
    if test_metrics:
        comparison_data['Dataset'].append('Test')
        comparison_data['Samples'].append(test_metrics['n_samples'])
        comparison_data['Accuracy'].append(test_metrics['accuracy'])
        comparison_data['Precision'].append(test_metrics['precision'])
        comparison_data['Recall'].append(test_metrics['recall'])
        comparison_data['F1 Score'].append(test_metrics['f1_score'])
        
        if test_metrics.get('roc_auc'):
            comparison_data['ROC AUC'].append(test_metrics['roc_auc'])
    
    df = pd.DataFrame(comparison_data)
    
    logger.info("\n" + df.to_string(index=False))
    
    # Overfitting kontrolü
    train_acc = train_metrics['accuracy']
    val_acc = val_metrics['accuracy']
    diff = train_acc - val_acc
    
    logger.info("\n" + "=" * 70)
    if diff < 0.05:
        logger.info("✅ Model dengeli (train-val fark < 0.05)")
    elif diff < 0.10:
        logger.warning("⚠️  Hafif overfitting (train-val fark 0.05-0.10)")
    else:
        logger.warning("❌ Overfitting var! (train-val fark > 0.10)")
    logger.info("=" * 70)