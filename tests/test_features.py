"""Feature pipeline: seasonal robust scaling and engineered features."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aiops_copilot.features.pipeline import FEATURE_COLUMNS, HostScaler, build_features, top_deviations
from conftest import Trained


def test_normal_rows_are_centred_near_zero(trained: Trained) -> None:
    test = trained.test
    z = trained.detector.scaler.transform(test)
    normal = test["is_anomaly"].to_numpy() == 0
    medians = np.median(np.abs(z[normal].to_numpy()), axis=0)
    # A robust z-score of normal traffic should mostly sit well inside +/-2.
    assert (medians < 2.0).all(), medians


def test_anomalous_rows_deviate_more_than_normal(trained: Trained) -> None:
    test = trained.test
    z = trained.detector.scaler.transform(test).abs().max(axis=1).to_numpy()
    anomalous = test["is_anomaly"].to_numpy() == 1
    assert np.median(z[anomalous]) > 2 * np.median(z[~anomalous])


def test_build_features_has_all_columns_and_no_nans(trained: Trained) -> None:
    feats = build_features(trained.test, trained.detector.scaler)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert not feats.isna().any().any()
    assert len(feats) == len(trained.test)


def test_scaler_roundtrip(trained: Trained, tmp_path: Path) -> None:
    path = tmp_path / "scaler.json"
    trained.detector.scaler.save(path)
    loaded = HostScaler.load(path)
    a = trained.detector.scaler.transform(trained.test.head(200))
    b = loaded.transform(trained.test.head(200))
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_top_deviations_sorted_by_magnitude(trained: Trained) -> None:
    z = trained.detector.scaler.transform(trained.test)
    top = top_deviations(z.iloc[0], k=3)
    assert len(top) == 3
    mags = [abs(v) for _, v in top]
    assert mags == sorted(mags, reverse=True)
