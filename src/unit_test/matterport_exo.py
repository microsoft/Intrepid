import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pdb

from skimage.transform import resize

# Read image
img = imageio.imread("./matterport_sample.png")
obs_dim = img.shape
print("Obs dim shape is ", obs_dim)

fnames = glob.glob("data/matterport/icon_figs/*png")
distractors = []

for fname in fnames:
    distractor_img = imageio.imread(fname)
    print("Read distractor from %s of size %r" % (fname, distractor_img.shape))

    assert len(distractor_img.shape) == 3 and (
        distractor_img.shape[2] == 3 or distractor_img.shape[2] == 4
    ), "Can only read RGB and RGBA images"
    if distractor_img.shape[2] == 4:
        distractor_img = distractor_img[:, :, :3]

    # Resize based on original image so that width of the obstacle is 10% of the width and
    # height is at most 40% of the height
    distractor_img = resize(
        distractor_img,
        (
            min(distractor_img.shape[0], int(0.2 * obs_dim[0])),
            min(distractor_img.shape[1], int(0.2 * obs_dim[1])),
            3,
        ),
    )
    distractor_img = (distractor_img * 255).astype(np.uint8)
    distractors.append(distractor_img)

print("Read %d many distractors " % len(distractors))

distractor_hor = 40
distractor_ver = 30
distractor_id = 0

# Add distractor
distractor_img = distractors[distractor_id]
distractor_shape = distractor_img.shape

img_slice = img[
    distractor_ver : distractor_ver + distractor_shape[0],
    distractor_hor : distractor_hor + distractor_shape[1],
    :,
]

print("Img shape is ", img.shape)
print("Img slice's shape is ", img_slice.shape)
print("Distractor slice's shape is ", distractor_shape)

distractor_img = distractor_img.reshape((-1, 3))
img_slice = img_slice.reshape((-1, 3))
distractor_img_min = distractor_img.min(1)
blue_pixel_ix = np.argwhere(distractor_img_min < 220)  # flattened (x, y) position where pixels are blue in color
values = np.squeeze(distractor_img[blue_pixel_ix])
np.put_along_axis(img_slice, blue_pixel_ix, values, axis=0)

img_slice = img_slice.reshape(distractor_shape)  # distractor and img_slice have the same shape

img[
    distractor_ver : distractor_ver + distractor_shape[0],
    distractor_hor : distractor_hor + distractor_shape[1],
    :,
] = img_slice

imgplot = plt.imshow(img)
plt.show()

pdb.set_trace()
