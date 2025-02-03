from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import MetricCollection


def get_metrics(**kwargs) -> MetricCollection:
    """
    Creates and returns a collection of metrics for evaluating model performance.

    Args:
        ``**kwargs``: Configuration options for initializing metrics.

    Returns:
        MetricCollection: A collection containing the Mean Average Precision (mAP) metric.
    """
    return MetricCollection(
        {
            'map': MeanAveragePrecision(**kwargs),
        },
    )
