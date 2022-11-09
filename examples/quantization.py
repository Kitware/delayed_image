"""
Demo for how to treat a uint8 0-255 image on disk as if it was a float32 0-1 image.
"""
import delayed_image
import kwimage

# Grab a test image
fpath = kwimage.grab_test_image_fpath('amazon')

# Point at the test image
raw = delayed_image.DelayedLoad(fpath)

# Call prepare to auto-populate dsize / channels. If these are given as
# arguments to DelayedLoad, then this is not necessary and will be a no-op.
raw = raw.prepare()

# Tell the delayed operations how to normalize this particular image
# Note: the "orig" values are the output, and the "quant" values are what
# exists on disk.
normalized = raw.dequantize({
    'orig_dtype': 'float32',
    'orig_min': 0,
    'orig_max': 1,
    'quant_min': 0,
    'quant_max': 255,
})

# Do whatever other operations you want on top of the tree.
final_tree = normalized[0:100, 40:300].scale(0.5)

# Take a peek at the operation tree
final_tree.write_network_text()

# Call finalize to get the raw ndarray at the end.
output_data = final_tree.finalize()
