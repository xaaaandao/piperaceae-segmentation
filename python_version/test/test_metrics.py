import pytest

from metrics import mean_all_metrics, sum_all_metrics


class TestMetrics:

    @pytest.fixture(autouse=True)
    def before_each(self):
        self.first_metric = {"iou_train": 10, "dice_train": 10, "iou_val": 10, "dice_val": 10, "iou_test": 10, "dice_test": 10}

    def test_sum_all_metrics(self):
        sum_all_metrics(self.first_metric, {"iou_train": 20, "dice_train": 20, "iou_val": 20, "dice_val": 20, "iou_test": 20, "dice_test": 20})
        assert True.__eq__(all(value == 30 for value in self.first_metric.values()))

    def test_mean_all_metrics(self):
        mean_all_metrics(self.first_metric, fold=10)
        assert True.__eq__(all(value == 1 for value in self.first_metric.values()))
