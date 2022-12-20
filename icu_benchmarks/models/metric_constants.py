from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    r2_score,
    mean_squared_error, f1_score
)

from ignite.contrib.metrics import (
    AveragePrecision,
    ROC_AUC,
    PrecisionRecallCurve,
    RocCurve
)

from ignite.metrics import MeanAbsoluteError, Accuracy, ConfusionMatrix

from icu_benchmarks.models.metrics import CalibrationCurve, BalancedAccuracy


class MLMetrics:
    BINARY_CLASSIFICATION = {
        "PR": average_precision_score,
        "AUC": roc_auc_score,
        "ROC": roc_curve,
        "PR": precision_recall_curve,
        "Calibration_Curve": calibration_curve,
        "Confusion_Matrix": confusion_matrix,
        "F1": f1_score
    }

    MULTICLASS_CLASSIFICATION = {
        "Accuracy": accuracy_score,
        "Balanced_Accuracy": balanced_accuracy_score,
        "PR": average_precision_score,
        "AUC": roc_auc_score,
        "Confusion_Matrix": confusion_matrix,
        "F1": f1_score,
    }

    REGRESSION = {
        "MAE": mean_absolute_error,
        "R2": r2_score,
        "RMSE": mean_squared_error
    }


class DLMetrics:
    BINARY_CLASSIFICATION = {
        "PR": AveragePrecision(),
        "AUC": ROC_AUC(),
        "PRC": PrecisionRecallCurve(),
        "ROC": RocCurve(),
        "Calibration_Curve": CalibrationCurve(),
        "Confusion_Matrix": ConfusionMatrix(num_classes=2),
    }

    MULTICLASS_CLASSIFICATION = {
        "Accuracy": Accuracy(),
        "BalancedAccuracy": BalancedAccuracy()
    }

    REGRESSION = {
        "MAE": MeanAbsoluteError()
    }


