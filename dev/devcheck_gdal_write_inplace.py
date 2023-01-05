"""
I'm in a situation where I have a large number of buffers representing frames
in a video open, and I'm stitching the results of an algorithm into them. When
a frame is finished it writes its output to disk as a COG.

As I move through the video, the number of unfinished frames I need to keep in
memory grows too large to fit in RAM. However, often an unfinished frame isn't
actually needed for several iterations. To work around the RAM issue, I'm
thinking of writing the unfinished frame to disk, and then reloading it when it
is needed again.

However, I was wondering if there is a way to do this more efficiently by
simply writing the impacted tiles / overviews in the COG? I see rasterio has
the ability to do windowed reading and writing, but it looks like when I do a
windowed write it ...

oh, if I use r+ mode, it seems to work
"""


import numpy as np
import kwimage
data = np.zeros((2048, 2048), dtype=np.uint8) + 1

fpath = 'foo.tif'

data = kwimage.ensure_uint255(np.random.rand(*data.shape))

kwimage.imwrite(fpath, data, backend='gdal', overviews=2, blocksize=256)


from delayed_image import lazy_loaders
self = lazy_loaders.LazyGDalFrameFile(fpath)
self[0:100, 20:200]
ds = self._ds
band = ds.GetRasterBand(1)

new = np.ones((32, 16), dtype=np.uint8)

# band.WriteArray(new, xoff=96, yoff=128)
new = data + 10
band.WriteArray(new)
band.FlushCache()
ds.FlushCache()

band.ReadAsArray().sum()

band = None
ds = None
self = None

before = kwimage.imread(fpath)


import rasterio
from rasterio.windows import Window

new = np.ones((32, 16), dtype=np.uint8) + 10
with rasterio.open(fpath, 'r+', driver='COG', width=2048, height=2048, count=1, dtype=np.uint8) as dst:
    dst.write(new, window=Window(2, 3, 16, 32), indexes=1)

after = kwimage.imread(fpath)
