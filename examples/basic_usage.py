"""
Basic usage demo.
"""
import kwimage
import delayed_image


def demo():
    import kwplot
    kwplot.autompl()

    # Lets say you have some image. Maybe it is a very large image, but maybe not.
    # Either way, you have a path to it.
    import kwimage
    fpath = kwimage.grab_test_image_fpath('amazon')
    print(f'fpath={fpath}')

    """
    ### The DelayedLoad object

    Perhaps there are a series of operations you want to perform on the image.
    To start create an instance of a ``DelayedLoad`` object.

    This simply stores the path to the image. We can also pass it extra
    information about the image if we know it a-priori, like the shape and
    number of / names of the channels.
    """

    # A simple reference to the underlying data
    delayed = delayed_image.DelayedLoad(fpath)

    print(delayed)

    # If you just wanted to just read all of the image data as-is, you can finalize
    # a DelayedImage at any time.
    imdata = delayed.finalize()
    kwplot.imshow(imdata)

    # Some operations do need to know  we need to know
    delayed.prepare()

