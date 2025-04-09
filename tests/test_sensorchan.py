import pytest


def test_sensorchan_streams_with_empty_chans():
    """
    Test corner case where a sensor exists, but it has no channels.
    """
    pytest.importorskip("lark")
    from delayed_image.sensorchan_spec import SensorChanSpec  # NOQA
    sensorchan = SensorChanSpec.coerce('sensor1:')
    assert sensorchan.numel() == 0
    assert sensorchan.chans.numel() == 0
    assert all(s.numel() == 0 for s in sensorchan.streams())
