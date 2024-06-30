from line_profiler import profile

profile.show_config['details'] = 0
profile.enable()

import ubelt as ub  # NOQA
from delayed_image import DelayedLoad  # NOQA


@profile
def benchmark_lots_of_objects():
    fpath = DelayedLoad.demo().fpath

    num_loads = 50_000

    with ub.Timer('making lots of objects'):
        for index in ub.ProgIter(range(num_loads), desc='iterating'):
            delayed = DelayedLoad(
                fpath, dsize=(512, 512), channels='red|green|blue')
            # delayed = delayed.prepare()
            delayed = delayed.crop(slice(None, None))
            delayed = delayed.warp({'scale': 2.0})
            delayed = delayed.crop(slice(None, None))
            # delayed.optimize()


def main():
    benchmark_lots_of_objects()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/delayed_image/dev/bench/benchmark_lots_of_objects.py
    """
    main()
