

def test_find_reference_scale():
    try:
        from rich import print as rprint
        use_rich = 1
    except ImportError:
        rprint = print
        use_rich = 0
    import delayed_image
    raw = delayed_image.DelayedLoad.demo(key='astro').prepare()

    import kwimage
    transform = kwimage.Affine.random()
    transform1 = kwimage.Affine.random()
    transform2 = kwimage.Affine.random()

    # Create a reference tranform, and a modification on top of that.
    # ref = raw.warp(transform)[0:255, 101:207]
    ref = raw.warp(transform)
    mod = ref.warp(transform1).warp(transform2)
    # [0:100, 0:100]

    opt_ref = ref.optimize()
    opt_mod = mod.optimize()

    rprint('\n-- [green] REF --')
    ref.write_network_text(rich=use_rich)

    rprint('\n-- [green] REF (opt) --')
    opt_ref.write_network_text(rich=use_rich)

    rprint('\n-- [red] MOD --')
    mod.write_network_text(rich=use_rich)

    rprint('\n-- [red] MOD (opt) --')
    opt_mod.write_network_text(rich=use_rich)

    # If we know we have a common leaf, then we should be able to construct
    # the transform from one image to the other.
    tf_ref_from_leaf = opt_ref.get_transform_from_leaf()
    tf_mod_from_leaf = opt_mod.get_transform_from_leaf()
    tf_leaf_from_ref = tf_ref_from_leaf.inv()

    tf_mod_from_ref = tf_mod_from_leaf @ tf_leaf_from_ref

    recon_mod = opt_ref.warp(tf_mod_from_ref, dsize=mod.dsize)
    recon_opt_mod = recon_mod.optimize()

    rprint('\n-- [orange1] MOD (recon) --')
    recon_mod.write_network_text(rich=use_rich)

    rprint('\n-- [orange1] MOD (recon, opt) --')
    recon_opt_mod.write_network_text(rich=use_rich)

    tf1 = recon_mod.get_transform_from_leaf()
    tf2 = mod.get_transform_from_leaf()
    tf3 = opt_mod.get_transform_from_leaf()

    # https://math.stackexchange.com/questions/507742/distance-similarity-between-two-matrices
    import numpy as np
    d12 = np.linalg.norm(tf1.matrix - tf2.matrix, ord='fro')
    d23 = np.linalg.norm(tf2.matrix - tf3.matrix, ord='fro')
    d13 = np.linalg.norm(tf1.matrix - tf3.matrix, ord='fro')
    assert np.isclose(d12, 0)
    assert np.isclose(d23, 0)
    assert np.isclose(d13, 0)

    tf_mod_from_ref2 = opt_mod.get_transform_from(opt_ref)
    d = np.linalg.norm(
        tf_mod_from_ref.matrix - tf_mod_from_ref2.matrix, ord='fro')
    assert np.isclose(d, 0)
