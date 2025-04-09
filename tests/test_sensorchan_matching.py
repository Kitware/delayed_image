from delayed_image.channel_spec import FusedChannelSpec  # NOQA
from delayed_image.channel_spec import ChannelSpec  # NOQA
from delayed_image.sensorchan_spec import SensorChanSpec  # NOQA
from delayed_image.sensorchan_spec import FusedSensorChanSpec  # NOQA
from delayed_image.sensorchan_spec import SensorSpec  # NOQA
import ubelt as ub
import pytest


@ub.memoize
def build_spec_variants():
    pytest.importorskip("lark")

    import itertools as it

    sensors = [
        'sensor1', 'sensor2'
    ]
    channels = [
        'r|g|b',
        'c|m|y|k',
        'gray',
    ]

    spec_variants = {
        'FusedChannelSpec': [],
        'ChannelSpec': [],
        'SensorChanSpec': [],
        'FusedSensorChanSpec': [],
    }
    for chan in channels:
        spec_variants['FusedChannelSpec'].append(FusedChannelSpec.coerce(chan))

    for k in range(1, len(channels) + 1):
        for chans in it.combinations(channels, k):
            spec = ','.join(chans)
            spec_variants['ChannelSpec'].append(ChannelSpec.coerce(spec).normalize().concise())

    # All possible ways to generate specs with the available sensors / channels
    for k in range(1, len(channels) + 1):
        for chans in it.combinations(channels, k):
            basis = {c: sensors for c in chans}
            for assignment in ub.named_product(basis):
                spec = ','.join([f'{s}:{c}' for c, s in assignment.items()])
                spec_variants['SensorChanSpec'].append(SensorChanSpec.coerce(spec).normalize().concise())

    for s in sensors:
        for c in channels:
            spec = f'{s}:{c}'
            spec_variants['FusedSensorChanSpec'].append(FusedSensorChanSpec.coerce(spec).normalize().concise())

    # print(f'spec_variants = {ub.urepr(spec_variants, nl=2)}')
    return spec_variants


def test_matching_sensor():
    """
    Ensure matching_sensor filters to the appropriate sensor
    """
    pytest.importorskip("lark")
    spec_variants = build_spec_variants()
    sensors = [
        'sensor1', 'sensor2'
    ]
    for sensorchan in spec_variants['FusedSensorChanSpec']:
        target_sensor = sensorchan.sensor.spec
        for sensor in sensors:
            result = sensorchan.matching_sensor(sensor)
            if sensor != target_sensor:
                assert result.chans.numel() == 0
            else:
                assert result.chans.numel() > 0

    for sensorchan in spec_variants['SensorChanSpec']:
        target_sensors = {s.sensor.spec for s in sensorchan.streams()}
        for sensor in sensors:
            result = sensorchan.matching_sensor(sensor)
            if sensor not in target_sensors:
                assert result.chans.numel() == 0
            else:
                assert result.chans.numel() > 0


def test_matching_sensor_generic():
    # generic sensor should match everything
    pytest.importorskip("lark")
    spec_variants = build_spec_variants()
    for sensorchan in spec_variants['FusedSensorChanSpec']:
        assert sensorchan.matching_sensor('*').numel() == sensorchan.numel()

    for sensorchan in spec_variants['SensorChanSpec']:
        assert sensorchan.matching_sensor('*').numel() == sensorchan.numel()


def test_empty_matching_sensor():
    """
    Test issue encounted in geowatch.
    """
    pytest.importorskip("lark")
    input_sensorchan = SensorChanSpec.coerce('sensor2:r|g|b,sensor3:r|g|b')
    matching = input_sensorchan.matching_sensor('sensor1')
    assert matching.numel() == 0
    assert matching.chans.numel() == 0
    assert all(s.numel() == 0 for s in matching.streams())
