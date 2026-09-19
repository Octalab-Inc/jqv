import os

import pytest


@pytest.fixture(scope="session")
def rt():
    from jqv.model import load_runtime

    return load_runtime(dtype=os.environ.get("JQV_TEST_DTYPE", "float32"))


@pytest.fixture(scope="session")
def bridge_items():
    from jqv.data import load_named

    return load_named("bridge")
