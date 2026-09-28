"""Keep runtime debug/episodic writes out of the repository during tests."""

import pytest
import graph_snapshots


@pytest.fixture(autouse=True)
def isolated_runtime_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_snapshots, 'ARTIFACT_DIRECTORY', tmp_path / 'episodic_memory')
