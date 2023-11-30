import delayed_image
import kwimage

# Grab a test image
fpath = kwimage.grab_test_image_fpath('amazon')

# Point at the test image
delayed = delayed_image.DelayedLoad(fpath).prepare()


# Crop to an area where part of it doesn't exist.
# By setting wrap and clip to False it should pretend you have an "infinite
# image" and finalize will pad the real data to match your request.
space_slice = (slice(-256, 256), slice(-256, 256))
cropped = delayed.crop(space_slice, wrap=False, clip=False)


# Inspect the operation tree
cropped.write_network_text()

imdata = cropped.finalize()
assert imdata.shape == (512, 512, 3)
