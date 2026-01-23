import pytest


@pytest.fixture(params=["legacy", "ast"])
def optimize_func(request):
    if request.param == "legacy":
        return lambda node, **kwargs: node.optimize(**kwargs)
    return lambda node, **kwargs: node.optimize_ast(**kwargs)


@pytest.fixture()
def optimize_pair():
    def _pair(node, **kwargs):
        return node.optimize(**kwargs), node.optimize_ast(**kwargs)
    return _pair
