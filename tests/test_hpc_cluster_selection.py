from __future__ import annotations

import pytest

from ropt.components.executors._hpc_executor import _select_cluster
from ropt.exceptions import ExecutionError


class _MockClusterAdapter:
    def __init__(self, clusters: dict[str, list[str]]) -> None:
        self._clusters = clusters
        self.current = next(iter(clusters), None)

    def list_clusters(self) -> list[str]:
        return list(self._clusters)

    def switch_cluster(self, cluster_name: str) -> None:
        if cluster_name not in self._clusters:
            raise KeyError(cluster_name)
        self.current = cluster_name

    @property
    def queue_list(self) -> list[str]:
        assert self.current is not None
        return self._clusters[self.current]


def test_cluster_derived_from_queue() -> None:
    adapter = _MockClusterAdapter({"cpu": ["short", "long"], "gpu": ["gpu_short"]})
    _select_cluster(adapter, None, "gpu_short")
    assert adapter.current == "gpu"


def test_queue_without_cluster() -> None:
    adapter = _MockClusterAdapter({"cpu": ["short"], "gpu": ["gpu_short"]})
    with pytest.raises(
        ExecutionError, match="Queue 'missing' is not available on any HPC cluster"
    ):
        _select_cluster(adapter, None, "missing")


def test_queue_in_multiple_clusters() -> None:
    adapter = _MockClusterAdapter({"cpu": ["shared"], "gpu": ["shared"]})
    with pytest.raises(
        ExecutionError, match="available on multiple HPC clusters: cpu, gpu"
    ):
        _select_cluster(adapter, None, "shared")


def test_explicit_cluster_without_queue() -> None:
    adapter = _MockClusterAdapter({"cpu": ["short"], "gpu": ["gpu_short"]})
    _select_cluster(adapter, "gpu", None)
    assert adapter.current == "gpu"


def test_explicit_cluster_with_queue() -> None:
    adapter = _MockClusterAdapter({"cpu": ["short"], "gpu": ["gpu_short"]})
    _select_cluster(adapter, "gpu", "gpu_short")
    assert adapter.current == "gpu"


def test_explicit_cluster_missing_queue() -> None:
    adapter = _MockClusterAdapter({"cpu": ["short"], "gpu": ["gpu_short"]})
    with pytest.raises(
        ExecutionError, match="Queue 'short' is not available on HPC cluster 'gpu'"
    ):
        _select_cluster(adapter, "gpu", "short")


def test_unknown_cluster() -> None:
    adapter = _MockClusterAdapter({"cpu": ["short"], "gpu": ["gpu_short"]})
    with pytest.raises(ExecutionError, match="Unknown HPC cluster: tpu"):
        _select_cluster(adapter, "tpu", None)
