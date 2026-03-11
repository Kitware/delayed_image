#!/usr/bin/env python3
"""
Benchmark delayed_image across multiple git refs and plot relative speedups.

Example:
    python dev/benchmarks/compare_branches.py compare --refs origin/main HEAD
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import time

SCRIPT_FPATH = pathlib.Path(__file__).resolve()
REPO_DPATH = SCRIPT_FPATH.parents[2]
BENCH_DPATH = REPO_DPATH / 'dev' / 'benchmarks'
CACHE_DPATH = BENCH_DPATH / '_cache'
RESULTS_DPATH = BENCH_DPATH / 'results'


def _slugify(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '-', text).strip('-').lower() or 'ref'


def _run(cmd, cwd=None, env=None, capture=False):
    cmd = [str(c) for c in cmd]
    if capture:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return None


def _git_output(args):
    return _run(['git', *args], cwd=REPO_DPATH, capture=True).strip()


def _is_working_tree_ref(ref: str) -> bool:
    return ref.lower() in {'working-tree', 'worktree', 'workspace', 'worktree+'}


def resolve_default_refs():
    candidates = [['origin/main', 'HEAD'], ['main', 'HEAD']]
    for refs in candidates:
        try:
            for ref in refs:
                _git_output(['rev-parse', '--verify', ref])
        except subprocess.CalledProcessError:
            continue
        return refs
    return ['HEAD']


def ensure_worktree(ref: str) -> tuple[str, pathlib.Path]:
    commit = _git_output(['rev-parse', ref])
    short = commit[:12]
    slug = _slugify(ref)
    worktree_dpath = CACHE_DPATH / 'worktrees' / f'{slug}-{short}'
    if not worktree_dpath.exists():
        worktree_dpath.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ['git', 'worktree', 'add', '--detach', str(worktree_dpath), commit],
            cwd=REPO_DPATH,
        )
    return commit, worktree_dpath


def resolve_ref_repo(ref: str) -> tuple[str, pathlib.Path]:
    if _is_working_tree_ref(ref):
        commit = _git_output(['rev-parse', 'HEAD'])
        return commit, REPO_DPATH
    return ensure_worktree(ref)


def requirements_fingerprint(include_gdal: bool = False) -> str:
    hasher = hashlib.sha256()
    req_paths = [
        REPO_DPATH / 'requirements' / 'runtime.txt',
        REPO_DPATH / 'requirements' / 'headless.txt',
        REPO_DPATH / 'requirements' / 'tests.txt',
    ]
    if include_gdal:
        req_paths.append(REPO_DPATH / 'requirements' / 'gdal.txt')
    for path in req_paths:
        hasher.update(path.read_bytes())
    hasher.update(sys.version.encode('utf8'))
    hasher.update(b'matplotlib>=3.8')
    hasher.update(str(bool(include_gdal)).encode('utf8'))
    return hasher.hexdigest()[:12]


def ensure_env(
    python_exe: str | None = None,
    force_reinstall: bool = False,
    include_gdal: bool = False,
) -> pathlib.Path:
    if python_exe is None:
        python_exe = sys.executable
    env_tag = f'py{sys.version_info.major}{sys.version_info.minor}-{requirements_fingerprint(include_gdal=include_gdal)}'
    env_dpath = CACHE_DPATH / 'venvs' / env_tag
    python_bin = env_dpath / 'bin' / 'python'
    if force_reinstall and env_dpath.exists():
        shutil.rmtree(env_dpath)
    if python_bin.exists():
        return python_bin

    env_dpath.parent.mkdir(parents=True, exist_ok=True)
    _run([python_exe, '-m', 'venv', str(env_dpath)])
    _run(
        [
            str(python_bin),
            '-m',
            'pip',
            'install',
            '-U',
            'pip',
            'wheel',
            'setuptools',
        ]
    )
    _run(
        [
            str(python_bin),
            '-m',
            'pip',
            'install',
            '-r',
            str(REPO_DPATH / 'requirements' / 'runtime.txt'),
            '-r',
            str(REPO_DPATH / 'requirements' / 'headless.txt'),
            '-r',
            str(REPO_DPATH / 'requirements' / 'tests.txt'),
            'matplotlib>=3.8',
        ]
    )
    if include_gdal:
        _run(
            [
                str(python_bin),
                '-m',
                'pip',
                'install',
                '-r',
                str(REPO_DPATH / 'requirements' / 'gdal.txt'),
            ]
        )
    return python_bin


def activate_repo(repo_dpath: pathlib.Path) -> None:
    repo_dpath = repo_dpath.resolve()
    script_repo = REPO_DPATH.resolve()

    new_sys_path = []
    for item in sys.path:
        probe = pathlib.Path(item or os.getcwd()).resolve()
        if probe == script_repo:
            continue
        new_sys_path.append(item)
    sys.path[:] = [str(repo_dpath), *new_sys_path]

    for key in list(sys.modules):
        if key == 'delayed_image' or key.startswith('delayed_image.'):
            sys.modules.pop(key, None)


def has_gdal() -> bool:
    return importlib.util.find_spec('osgeo') is not None


def _channel_spec(num_channels: int) -> str:
    if num_channels == 3:
        return 'r|g|b'
    return '|'.join(f'c{i}' for i in range(num_channels))


def _synth_raster(shape, num_channels, dtype_name, phase=0.0):
    import numpy as np

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    chans = []
    for idx in range(num_channels):
        ch = (
            np.sin((xx + idx * 11 + phase * 3.0) / (17.0 + idx))
            + np.cos((yy + idx * 7 + phase * 5.0) / (23.0 + idx))
            + ((xx * (idx + 3) + yy * (idx + 5) + phase * 11.0) % (29 + idx))
            / (29.0 + idx)
        )
        chans.append(ch)
    data = np.stack(chans, axis=2)
    if dtype_name == 'uint8':
        data = data - data.min()
        data = data / max(float(data.max()), 1e-9)
        data = np.round(data * 255.0).astype(np.uint8)
    elif dtype_name == 'float32':
        data = data.astype(np.float32)
    else:
        raise KeyError(dtype_name)
    return data


def ensure_gdal_benchmark_raster(
    name, shape, num_channels, dtype_name, phase=0.0
):
    import kwimage

    data_dpath = CACHE_DPATH / 'data' / 'gdal'
    data_dpath.mkdir(parents=True, exist_ok=True)
    fpath = data_dpath / f'{name}.tif'
    if fpath.exists():
        return fpath

    data = _synth_raster(
        shape=shape,
        num_channels=num_channels,
        dtype_name=dtype_name,
        phase=phase,
    )
    tmp_fpath = data_dpath / f'{name}.tmp.tif'
    kwimage.imwrite(
        tmp_fpath,
        data,
        backend='gdal',
        overviews=4,
        blocksize=256,
        compress='DEFLATE',
    )

    try:
        from osgeo import gdal

        if hasattr(gdal, 'UseExceptions'):
            gdal.UseExceptions()
        cog_driver = gdal.GetDriverByName('COG')
    except Exception:
        cog_driver = None

    if cog_driver is not None:
        from osgeo import gdal

        gdal.Translate(
            str(fpath),
            str(tmp_fpath),
            format='COG',
            creationOptions=[
                'COMPRESS=DEFLATE',
                'BLOCKSIZE=256',
                'OVERVIEWS=AUTO',
            ],
        )
        tmp_fpath.unlink(missing_ok=True)
    else:
        tmp_fpath.rename(fpath)
    return fpath


def build_gdal_benchmark_cases(delayed_image, kwimage, np):
    if not has_gdal():
        raise RuntimeError(
            'GDAL benchmark cases requested, but osgeo.gdal is unavailable'
        )

    patch_specs = []
    rng = np.random.default_rng(101)
    for _ in range(4):
        x0 = int(rng.integers(96, 2048 - 96 - 1024))
        y0 = int(rng.integers(96, 2048 - 96 - 1024))
        theta = float(rng.uniform(-0.2, 0.2))
        patch_specs.append((x0, y0, theta))

    configs = [
        {
            'dtype_name': 'uint8',
            'num_channels': 3,
            'shape': (2048, 2048),
            'tag': 'u8_rgb',
        },
        {
            'dtype_name': 'uint8',
            'num_channels': 8,
            'shape': (2048, 2048),
            'tag': 'u8_multi',
        },
        {
            'dtype_name': 'float32',
            'num_channels': 3,
            'shape': (2048, 2048),
            'tag': 'f32_rgb',
        },
        {
            'dtype_name': 'float32',
            'num_channels': 8,
            'shape': (2048, 2048),
            'tag': 'f32_multi',
        },
    ]

    cases = []
    dst_dsize = (256, 256)
    crop_size = 1024
    scale = dst_dsize[0] / crop_size
    manyfiles_count = 4

    for config in configs:
        fpath = ensure_gdal_benchmark_raster(
            name=f'bench_{config["tag"]}',
            shape=config['shape'],
            num_channels=config['num_channels'],
            dtype_name=config['dtype_name'],
        )
        channels = _channel_spec(config['num_channels'])
        base = delayed_image.DelayedLoad(
            fpath,
            channels=channels,
            nodata_method='float',
        ).prepare()
        manyfile_specs = []
        for file_idx in range(manyfiles_count):
            manyfile_fpath = ensure_gdal_benchmark_raster(
                name=f'bench_manyfiles_{config["tag"]}_{file_idx:02d}',
                shape=config['shape'],
                num_channels=config['num_channels'],
                dtype_name=config['dtype_name'],
                phase=file_idx + 1,
            )
            x0 = int(rng.integers(96, 2048 - 96 - 1024))
            y0 = int(rng.integers(96, 2048 - 96 - 1024))
            theta = float(rng.uniform(-0.2, 0.2))
            manyfile_specs.append((manyfile_fpath, x0, y0, theta))

        for with_rotation in [False, True]:
            label = 'rotate' if with_rotation else 'no_rotate'

            def scenario(
                base=base, patch_specs=patch_specs, with_rotation=with_rotation
            ):
                total = 0.0
                crop_center = (crop_size / 2.0, crop_size / 2.0)
                dst_center = (dst_dsize[0] / 2.0, dst_dsize[1] / 2.0)
                for x0, y0, theta in patch_specs:
                    node = base.crop(
                        (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
                    )
                    warp = (
                        kwimage.Affine.translate(dst_center)
                        @ kwimage.Affine.rotate(theta if with_rotation else 0.0)
                        @ kwimage.Affine.scale(scale)
                        @ kwimage.Affine.translate(
                            (-crop_center[0], -crop_center[1])
                        )
                    )
                    node = node.warp(
                        warp,
                        dsize=dst_dsize,
                        interpolation='linear',
                        antialias=True,
                        border_value=0,
                    )
                    data = node.finalize(nodata_method='float')
                    total += float(np.nanmean(data))
                return total

            cases.append(
                {
                    'name': f'gdal_patch_{config["tag"]}_{label}',
                    'group': 'gdal_patch',
                    'description': (
                        f'GDAL/COG-style random patch sampling from {config["dtype_name"]} '
                        f'{config["num_channels"]}-channel imagery with {label.replace("_", " ")}.'
                    ),
                    'fn': scenario,
                }
            )

            def manyfiles_scenario(
                channels=channels,
                manyfile_specs=manyfile_specs,
                with_rotation=with_rotation,
            ):
                total = 0.0
                crop_center = (crop_size / 2.0, crop_size / 2.0)
                dst_center = (dst_dsize[0] / 2.0, dst_dsize[1] / 2.0)
                for fpath, x0, y0, theta in manyfile_specs:
                    base = delayed_image.DelayedLoad(
                        fpath,
                        channels=channels,
                        nodata_method='float',
                    ).prepare()
                    node = base.crop(
                        (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
                    )
                    warp = (
                        kwimage.Affine.translate(dst_center)
                        @ kwimage.Affine.rotate(theta if with_rotation else 0.0)
                        @ kwimage.Affine.scale(scale)
                        @ kwimage.Affine.translate(
                            (-crop_center[0], -crop_center[1])
                        )
                    )
                    node = node.warp(
                        warp,
                        dsize=dst_dsize,
                        interpolation='linear',
                        antialias=True,
                        border_value=0,
                    )
                    data = node.finalize(nodata_method='float')
                    total += float(np.nanmean(data))
                return total

            cases.append(
                {
                    'name': f'gdal_manyfiles_{config["tag"]}_{label}',
                    'group': 'gdal_manyfiles',
                    'description': (
                        f'GDAL/COG-style random patch sampling that cycles across {manyfiles_count} '
                        f'different {config["dtype_name"]} {config["num_channels"]}-channel images '
                        f'with {label.replace("_", " ")}.'
                    ),
                    'fn': manyfiles_scenario,
                }
            )
    return cases


def build_benchmark_cases(include_gdal: bool = False):
    import kwimage
    import numpy as np

    import delayed_image

    rng = np.random.default_rng(0)
    raw_gray = np.linspace(0, 1, 512 * 512, dtype=np.float32).reshape(512, 512)
    raw_rgb = rng.random((512, 512, 3), dtype=np.float32)
    raw_big = rng.random((768, 768, 3), dtype=np.float32)

    def construct_many_nodes():
        delayed = None
        for idx in range(400):
            delayed = delayed_image.DelayedIdentity(raw_rgb)
            delayed = delayed.crop((slice(idx % 7, 400), slice(idx % 5, 420)))
            delayed = delayed.warp(kwimage.Affine.scale(1.0 + (idx % 3) * 0.05))
            delayed = delayed.crop((slice(0, 256), slice(0, 256)))
        assert delayed is not None
        return delayed.dsize

    def optimize_deep_crop_chain():
        delayed = delayed_image.DelayedIdentity(raw_big)
        for idx in range(18):
            pad = idx % 4
            delayed = delayed.crop(
                (slice(pad, 720 - pad), slice(pad, 720 - pad))
            )
            delayed = delayed.warp(
                kwimage.Affine.scale(1.01 + (idx % 5) * 0.01)
                @ kwimage.Affine.translate((0.25, 0.5))
            )
        return delayed.optimize().shape

    def optimize_channel_stack():
        base = delayed_image.DelayedIdentity(raw_rgb)
        parts = []
        for chan_idx in range(3):
            part = base.take_channels([chan_idx])
            part = part.warp(kwimage.Affine.scale(1.2))
            part = part.crop((slice(5, 280), slice(7, 300)))
            parts.append(part)
        delayed = delayed_image.DelayedChannelConcat(parts)
        return delayed.optimize().shape

    def finalize_nearest_integer_scale():
        delayed = delayed_image.DelayedIdentity(raw_gray)
        warped = delayed.warp(kwimage.Affine.scale(2.0))
        data = warped.finalize(interpolation='nearest')
        return float(np.nanmean(data))

    def finalize_nearest_noninteger_scale():
        delayed = delayed_image.DelayedIdentity(raw_gray[:128, :128])
        warped = delayed.warp(
            kwimage.Affine.coerce(offset=(0, 0), scale=(8.6, 8.5))
        )
        data = warped.finalize(interpolation='nearest')
        return float(np.nanmean(data))

    def finalize_linear_affine_color():
        delayed = delayed_image.DelayedIdentity(raw_rgb)
        warp = (
            kwimage.Affine.translate((-10, 13))
            @ kwimage.Affine.rotate(math.pi / 16)
            @ kwimage.Affine.scale(0.85)
        )
        warped = delayed.warp(warp)
        data = warped.finalize(interpolation='linear')
        return float(data.mean())

    def finalize_optimized_pipeline():
        delayed = delayed_image.DelayedIdentity(raw_big)
        delayed = delayed.warp(kwimage.Affine.scale(0.75))
        delayed = delayed.crop((slice(15, 420), slice(10, 430)))
        delayed = delayed.warp(
            kwimage.Affine.translate((2.5, -3.5)) @ kwimage.Affine.scale(1.35)
        )
        delayed = delayed.crop((slice(0, 320), slice(0, 320)))
        delayed = delayed.take_channels([0, 1])
        data = delayed.finalize(interpolation='linear', nodata_method='float')
        return float(np.nanmean(data))

    cases = [
        {
            'name': 'construct_many_nodes',
            'group': 'construction',
            'description': 'Create many small delayed graphs without finalizing.',
            'fn': construct_many_nodes,
        },
        {
            'name': 'optimize_deep_crop_chain',
            'group': 'optimize',
            'description': 'Optimize a deep crop and warp pipeline.',
            'fn': optimize_deep_crop_chain,
        },
        {
            'name': 'optimize_channel_stack',
            'group': 'optimize',
            'description': 'Optimize a channel concat graph built from per-channel branches.',
            'fn': optimize_channel_stack,
        },
        {
            'name': 'finalize_nearest_integer_scale',
            'group': 'finalize',
            'description': 'Finalize a pure nearest-neighbor upscale.',
            'fn': finalize_nearest_integer_scale,
        },
        {
            'name': 'finalize_nearest_noninteger_scale',
            'group': 'finalize',
            'description': 'Finalize a non-integer nearest-neighbor warp.',
            'fn': finalize_nearest_noninteger_scale,
        },
        {
            'name': 'finalize_linear_affine_color',
            'group': 'finalize',
            'description': 'Finalize a color affine warp with linear interpolation.',
            'fn': finalize_linear_affine_color,
        },
        {
            'name': 'finalize_optimized_pipeline',
            'group': 'end_to_end',
            'description': 'Build and finalize a realistic multi-op image pipeline.',
            'fn': finalize_optimized_pipeline,
        },
    ]
    if include_gdal:
        cases.extend(build_gdal_benchmark_cases(delayed_image, kwimage, np))
    return cases


def time_one(fn, loops: int) -> float:
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(loops):
            fn()
        end = time.perf_counter()
    finally:
        if gc_was_enabled:
            gc.enable()
    return end - start


def benchmark_case(case, warmup: int, repeats: int, min_time: float) -> dict:
    fn = case['fn']
    for _ in range(warmup):
        fn()

    loops = 1
    elapsed = time_one(fn, loops)
    while elapsed < min_time and loops < (1 << 20):
        loops *= 2
        elapsed = time_one(fn, loops)

    samples = []
    for _ in range(repeats):
        gc.collect()
        elapsed = time_one(fn, loops)
        samples.append(elapsed / loops)

    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    mad = statistics.median(abs(item - median) for item in samples)
    return {
        'name': case['name'],
        'group': case['group'],
        'description': case['description'],
        'loops': loops,
        'samples_sec': samples,
        'median_sec': median,
        'mean_sec': mean,
        'stdev_sec': stdev,
        'mad_sec': mad,
    }


def suite_main(args):
    repo_dpath = pathlib.Path(args.repo).resolve()
    activate_repo(repo_dpath)

    commit = _run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo_dpath, capture=True
    ).strip()
    branch = _run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=repo_dpath,
        capture=True,
    ).strip()

    cases = build_benchmark_cases(include_gdal=args.include_gdal)
    results = [
        benchmark_case(
            case,
            warmup=args.warmup,
            repeats=args.repeats,
            min_time=args.min_time,
        )
        for case in cases
    ]
    payload = {
        'repo': str(repo_dpath),
        'branch': branch,
        'commit': commit,
        'python': sys.version,
        'warmup': args.warmup,
        'repeats': args.repeats,
        'min_time': args.min_time,
        'include_gdal': args.include_gdal,
        'cases': results,
    }

    output_fpath = pathlib.Path(args.output)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                'branch': branch,
                'commit': commit[:12],
                'output': str(output_fpath),
            },
            indent=2,
        )
    )


def load_json(path):
    return json.loads(pathlib.Path(path).read_text())


def plot_results(
    payloads, output_fpath: pathlib.Path, baseline_ref: str
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    baseline = payloads[baseline_ref]
    case_names = [case['name'] for case in baseline['cases']]
    label_map = {
        case['name']: case['description'] for case in baseline['cases']
    }
    baseline_times = {
        case['name']: case['median_sec'] for case in baseline['cases']
    }

    other_refs = [ref for ref in payloads if ref != baseline_ref]
    if not other_refs:
        return

    fig_height = max(4.5, 0.75 * len(case_names))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bar_height = 0.8 / max(len(other_refs), 1)
    centers = list(range(len(case_names)))

    for ref_idx, ref in enumerate(other_refs):
        case_map = {case['name']: case for case in payloads[ref]['cases']}
        offsets = [c - 0.4 + (ref_idx + 0.5) * bar_height for c in centers]
        speedups = [
            baseline_times[name] / case_map[name]['median_sec']
            for name in case_names
        ]
        ax.barh(offsets, speedups, height=bar_height, label=ref)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_yticks(centers)
    ax.set_yticklabels(case_names)
    ax.set_xlabel(f'Speedup vs {baseline_ref} (higher is faster)')
    ax.set_title('Delayed Image Benchmark Comparison')
    ax.legend()
    for idx, name in enumerate(case_names):
        ax.text(
            0.01,
            idx,
            textwrap.shorten(label_map[name], width=64, placeholder='...'),
            va='center',
            ha='left',
            fontsize=8,
            color='#333333',
            transform=ax.get_yaxis_transform(),
        )
    fig.tight_layout()
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fpath, dpi=200, bbox_inches='tight')
    plt.close(fig)


def summarize(payloads, baseline_ref: str) -> str:
    baseline = payloads[baseline_ref]
    baseline_map = {case['name']: case for case in baseline['cases']}
    lines = []
    for ref, payload in payloads.items():
        if ref == baseline_ref:
            continue
        lines.append(f'[{ref}] vs [{baseline_ref}]')
        for case in payload['cases']:
            base = baseline_map[case['name']]
            speedup = base['median_sec'] / case['median_sec']
            rel = (case['median_sec'] / base['median_sec']) - 1.0
            lines.append(
                f'  {case["name"]}: '
                f'median={case["median_sec"]:.6f}s '
                f'baseline={base["median_sec"]:.6f}s '
                f'speedup={speedup:.3f}x '
                f'rel={rel:+.1%}'
            )
    return '\n'.join(lines)


def plot_main(args):
    payload = load_json(args.input)
    plot_results(
        payload['payloads'],
        pathlib.Path(args.output),
        baseline_ref=args.baseline_ref,
    )
    print(args.output)


def compare_main(args):
    python_bin = ensure_env(
        python_exe=args.python,
        force_reinstall=args.force_reinstall,
        include_gdal=args.include_gdal,
    )
    if (
        pathlib.Path(sys.executable).resolve() != python_bin.resolve()
        and os.environ.get('DELAYED_IMAGE_BENCH_REEXEC') != '1'
    ):
        env = os.environ.copy()
        env['DELAYED_IMAGE_BENCH_REEXEC'] = '1'
        _run(
            [str(python_bin), str(SCRIPT_FPATH), *sys.argv[1:]],
            cwd=REPO_DPATH,
            env=env,
        )
        return

    refs = args.refs or resolve_default_refs()
    baseline_ref = args.baseline_ref or refs[0]

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_dpath = RESULTS_DPATH / f'compare-{timestamp}'
    run_dpath.mkdir(parents=True, exist_ok=True)

    payloads = {}
    manifest = {
        'refs': refs,
        'baseline_ref': baseline_ref,
        'python': str(python_bin),
        'results': {},
    }

    for ref in refs:
        commit, worktree_dpath = resolve_ref_repo(ref)
        out_fpath = run_dpath / f'{_slugify(ref)}-{commit[:12]}.json'
        cmd = [
            str(python_bin),
            str(SCRIPT_FPATH),
            'suite',
            '--repo',
            str(worktree_dpath),
            '--output',
            str(out_fpath),
            '--warmup',
            str(args.warmup),
            '--repeats',
            str(args.repeats),
            '--min-time',
            str(args.min_time),
        ]
        if args.include_gdal:
            cmd.append('--include-gdal')
        _run(cmd, cwd=REPO_DPATH)
        payload = load_json(out_fpath)
        payloads[ref] = payload
        manifest['results'][ref] = {
            'commit': commit,
            'worktree': str(worktree_dpath),
            'json': str(out_fpath),
        }

    comparison_fpath = run_dpath / 'comparison.json'
    comparison_fpath.write_text(
        json.dumps(
            {
                'manifest': manifest,
                'payloads': payloads,
            },
            indent=2,
        )
    )

    plot_fpath = run_dpath / 'speedup_histogram.png'
    _run(
        [
            str(python_bin),
            str(SCRIPT_FPATH),
            'plot',
            '--input',
            str(comparison_fpath),
            '--output',
            str(plot_fpath),
            '--baseline-ref',
            baseline_ref,
        ],
        cwd=REPO_DPATH,
    )

    print(f'results: {run_dpath}')
    print(f'plot:    {plot_fpath}')
    print(summarize(payloads, baseline_ref=baseline_ref))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    compare = subparsers.add_parser('compare', help='compare multiple refs')
    compare.add_argument(
        '--refs', nargs='+', default=None, help='git refs to compare'
    )
    compare.add_argument(
        '--baseline-ref', default=None, help='baseline ref for speedup ratios'
    )
    compare.add_argument(
        '--python',
        default=None,
        help='python executable used to build the benchmark venv',
    )
    compare.add_argument('--warmup', type=int, default=2)
    compare.add_argument('--repeats', type=int, default=9)
    compare.add_argument('--min-time', type=float, default=0.12)
    compare.add_argument('--force-reinstall', action='store_true')
    compare.add_argument(
        '--include-gdal',
        action='store_true',
        help='install GDAL and include COG patch benchmarks',
    )
    compare.set_defaults(main=compare_main)

    suite = subparsers.add_parser(
        'suite', help='run the benchmark suite for one repo'
    )
    suite.add_argument(
        '--repo', required=True, help='path to the target repo/worktree'
    )
    suite.add_argument('--output', required=True, help='json output path')
    suite.add_argument('--warmup', type=int, default=2)
    suite.add_argument('--repeats', type=int, default=9)
    suite.add_argument('--min-time', type=float, default=0.12)
    suite.add_argument(
        '--include-gdal',
        action='store_true',
        help='include GDAL/COG patch benchmarks',
    )
    suite.set_defaults(main=suite_main)

    plot = subparsers.add_parser(
        'plot', help='plot a completed comparison json'
    )
    plot.add_argument('--input', required=True, help='comparison json path')
    plot.add_argument('--output', required=True, help='png output path')
    plot.add_argument('--baseline-ref', required=True, help='baseline ref key')
    plot.set_defaults(main=plot_main)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.main(args)


if __name__ == '__main__':
    main()
