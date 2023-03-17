from argparse import Namespace
import torch
import h5py
from PIL import Image
from glob import glob
from torchvision import transforms
from matplotlib import pyplot as plt
from object_discovery.method import SlotAttentionMethod
from object_discovery.slot_attention_model import SlotAttentionModel
from object_discovery.utils import slightly_off_center_crop


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path)
    params = Namespace(**ckpt["hyper_parameters"])
    sa = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        slot_size=params.slot_size,
    )
    model = SlotAttentionMethod.load_from_checkpoint(
        ckpt_path, model=sa, datamodule=None
    )
    model.eval()
    return model


print("Loading model...")
ckpt_path = "epoch=209-step=299460.ckpt"
model = load_model(ckpt_path)

t = transforms.ToTensor()

print("Loading images...")
with h5py.File("data/box_world_dataset.h5", "r") as f:
    images = f["image"][0:8]

transformed_images = []
for image in images:
    transformed_images.append(t(image))
images = torch.stack(transformed_images)

print("Predicting...")
images = model.predict(
    images,
    do_transforms=True,
    debug=True,
    return_pil=True,
    background_detection="both",
)
slots = model.predict(images, do_transforms=True, return_slots=True)
slots = slots.squeeze()
# `slots` has shape (num_slots, num_features)

print("Saving...")
images.save("output.png")


# ckpt_path = "sketchy_sa-epoch=59-step=316440-3nofluv3.ckpt"
# model = load_model(ckpt_path)

# transformed_images = []
# for image_path in glob("data/sketchy_sample/*.png"):
#     image = Image.open(image_path)
#     image = image.convert("RGB")
#     transformed_images.append(transforms.functional.to_tensor(image))

# images = model.predict(
#     torch.stack(transformed_images),
#     do_transforms=True,
#     debug=True,
#     return_pil=True,
#     background_detection="both",
# )

# plt.imshow(images, interpolation="nearest")
# plt.show()

# ckpt_path = "clevr6_masks-epoch=673-step=275666-r4nbi6n7.ckpt"
# model = load_model(ckpt_path)

# t = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Lambda(slightly_off_center_crop),
#     ]
# )

# with h5py.File("/media/Main/Downloads/clevr_with_masks.h5", "r") as f:
#     images = f["image"][0:8]

# transformed_images = []
# for image in images:
#     transformed_images.append(t(image))
# images = torch.stack(transformed_images)

# images = model.predict(
#     images,
#     do_transforms=True,
#     debug=True,
#     return_pil=True,
#     background_detection="both",
# )

# plt.imshow(images, interpolation="nearest")
# plt.show()
