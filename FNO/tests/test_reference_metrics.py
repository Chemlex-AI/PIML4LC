import numpy as np
import pytest
from src.verify import curve_scores


def test_reference_scores_use_peak_time_and_exclude_missing_peak():
    target = np.zeros((10, 3))
    target[3, 0] = 1
    target[6, 1] = 2
    prediction = np.zeros_like(target)
    prediction[4, 0] = 1
    prediction[4, 1] = 2
    result = curve_scores(prediction, target)
    assert result['included_experiments'] == [0, 1]
    assert result['peak_time_difference_s'] == [1, -2]
    assert result['retention_time_rmse_s'] == pytest.approx(np.sqrt(2.5))
