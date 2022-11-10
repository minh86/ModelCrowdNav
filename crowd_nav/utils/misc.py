from PIL import Image, ImageSequence


def Resize_GIF(src_path):
    # Output (max) size
    size = 800, 600
    # Open source
    im = Image.open(src_path)
    # Get sequence iterator
    frames = ImageSequence.Iterator(im)

    # Wrap on-the-fly thumbnail generator
    def thumbnails(frames):
        for frame in frames:
            thumbnail = frame.copy()
            thumbnail.thumbnail(size, Image.ANTIALIAS)
            yield thumbnail

    frames = thumbnails(frames)
    # Save output
    om = next(frames)  # Handle first frame separately
    om.info = im.info  # Copy sequence info
    om.save(src_path, save_all=True, append_images=list(frames))

def PositiveRate(memory):
    pos = 0
    for _, value in memory.memory:
        if value.item() >0:
            pos+=1
    return  pos/len(memory.memory)