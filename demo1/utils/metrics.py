import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score


def calculate_metrics(y_true, y_pred):
    """
    计算分类评估指标

    参数:
    y_true: 真实标签
    y_pred: 预测标签

    返回:
    包含各种评估指标的字典
    """
    # 确保输入是一维数组
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # 只考虑有效像素（非零标签）
    valid_idx = y_true > 0
    if np.sum(valid_idx) > 0:
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]

    # 计算混淆矩阵
    class_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # 计算整体准确率
    oa = accuracy_score(y_true, y_pred)

    # 计算每类准确率
    per_class_acc = np.zeros(len(class_labels))
    for i, label in enumerate(class_labels):
        class_idx = y_true == label
        if np.sum(class_idx) > 0:
            per_class_acc[i] = accuracy_score(y_true[class_idx], y_pred[class_idx])

    # 计算平均准确率
    aa = np.mean(per_class_acc)

    # 计算Kappa系数
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        'confusion_matrix': cm,
        'overall_accuracy': oa,
        'average_accuracy': aa,
        'per_class_accuracy': per_class_acc,
        'kappa': kappa,
        'class_labels': class_labels
    }