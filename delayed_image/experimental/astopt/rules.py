"""Rule registry for the experimental AST optimizer."""

from __future__ import __annotations__

from collections import defaultdict

import ubelt as ub

from delayed_image import delayed_nodes


RULES = defaultdict(list)


def register_rule(node_type):
    def decorator(func):
        RULES[node_type].append(func)
        return func
    return decorator


def rules_for(node):
    rules = []
    for cls in node.__class__.__mro__:
        rules.extend(RULES.get(cls, []))
    return rules


DelayedWarp = delayed_nodes.DelayedWarp
DelayedCrop = delayed_nodes.DelayedCrop
DelayedOverview = delayed_nodes.DelayedOverview
DelayedDequantize = delayed_nodes.DelayedDequantize
DelayedChannelConcat = delayed_nodes.DelayedChannelConcat
isinstance2 = delayed_nodes.isinstance2


@register_rule(DelayedWarp)
def fuse_warps(node):
    if isinstance2(node.subdata, DelayedWarp):
        return node._opt_fuse_warps(), True, "fuse_warps"
    return node, False, None


@register_rule(DelayedWarp)
def remove_identity_warp(node):
    noop_eps = node.meta.get("noop_eps", 0)
    is_negligible = (
        node.dsize == node.subdata.dsize
        and node.transform.isclose_identity(rtol=noop_eps, atol=noop_eps)
    )
    if is_negligible:
        return node.subdata, True, "remove_identity_warp"
    return node, False, None


@register_rule(DelayedWarp)
def push_warp_under_concat(node):
    if isinstance2(node.subdata, DelayedChannelConcat):
        return node._opt_push_under_concat(), True, "warp_under_concat"
    return node, False, None


@register_rule(DelayedWarp)
def warp_on_optimized_subdata(node):
    if hasattr(node.subdata, "_optimized_warp"):
        warp_kwargs = ub.dict_isect(node.meta, node._data_keys + node._algo_keys)
        return node.subdata._optimized_warp(**warp_kwargs), True, "subdata_optimized_warp"
    return node, False, None


@register_rule(DelayedWarp)
def split_warp_overview(node):
    split = node._opt_split_warp_overview()
    if split is not node:
        return split, True, "split_warp_overview"
    return node, False, None


@register_rule(DelayedWarp)
def absorb_overview(node):
    absorbed = node._opt_absorb_overview()
    if absorbed is not node:
        return absorbed, True, "absorb_overview"
    return node, False, None


@register_rule(DelayedCrop)
def fuse_crops(node):
    if isinstance2(node.subdata, DelayedCrop):
        return node._opt_fuse_crops(), True, "fuse_crops"
    return node, False, None


@register_rule(DelayedCrop)
def optimized_crop_subdata(node):
    if hasattr(node.subdata, "_optimized_crop"):
        crop_kwargs = ub.dict_isect(node.meta, {"space_slice", "chan_idxs"})
        return node.subdata._optimized_crop(**crop_kwargs), True, "subdata_optimized_crop"
    return node, False, None


@register_rule(DelayedCrop)
def crop_after_warp(node):
    if isinstance2(node.subdata, DelayedWarp):
        return node._opt_warp_after_crop(), True, "crop_after_warp"
    return node, False, None


@register_rule(DelayedCrop)
def dequant_after_crop(node):
    if isinstance2(node.subdata, DelayedDequantize):
        return node._opt_dequant_after_crop(), True, "dequant_after_crop"
    return node, False, None


@register_rule(DelayedCrop)
def crop_under_concat(node):
    if isinstance2(node.subdata, DelayedChannelConcat):
        if node.meta.get("chan_idxs", None) is not None:
            return node, False, None
        return node._opt_push_under_concat(), True, "crop_under_concat"
    return node, False, None


@register_rule(DelayedOverview)
def fuse_overview(node):
    if isinstance2(node.subdata, DelayedOverview):
        return node._opt_fuse_overview(), True, "fuse_overview"
    return node, False, None


@register_rule(DelayedOverview)
def drop_identity_overview(node):
    if node.meta.get("overview", None) == 0:
        return node.subdata, True, "drop_overview_0"
    return node, False, None


@register_rule(DelayedOverview)
def crop_after_overview(node):
    if isinstance2(node.subdata, DelayedCrop):
        return node._opt_crop_after_overview(), True, "crop_after_overview"
    return node, False, None


@register_rule(DelayedOverview)
def warp_after_overview(node):
    if isinstance2(node.subdata, DelayedWarp):
        return node._opt_warp_after_overview(), True, "warp_after_overview"
    return node, False, None


@register_rule(DelayedOverview)
def dequant_after_overview(node):
    if isinstance2(node.subdata, DelayedDequantize):
        return node._opt_dequant_after_overview(), True, "dequant_after_overview"
    return node, False, None


@register_rule(DelayedOverview)
def overview_under_concat(node):
    if isinstance2(node.subdata, DelayedChannelConcat):
        return node._opt_push_under_concat(), True, "overview_under_concat"
    return node, False, None


@register_rule(DelayedDequantize)
def dequant_before_warp(node):
    if isinstance2(node.subdata, DelayedWarp):
        return node._opt_dequant_before_other(), True, "dequant_before_warp"
    return node, False, None


@register_rule(DelayedDequantize)
def dequant_under_concat(node):
    if isinstance2(node.subdata, DelayedChannelConcat):
        return node._opt_push_under_concat(), True, "dequant_under_concat"
    return node, False, None
