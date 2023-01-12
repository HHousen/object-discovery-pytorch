from argparse import Namespace
import torch
import h5py
from torchvision import transforms
from matplotlib import pyplot as plt
from object_discovery.method import SlotAttentionMethod
from object_discovery.slot_attention_model import SlotAttentionModel
from object_discovery.utils import slightly_off_center_crop


CKPT_PATH = "clevr6_masks-epoch=673-step=275666-r4nbi6n7.ckpt"

ckpt = torch.load(CKPT_PATH)
params = Namespace(**ckpt["hyper_parameters"])
sa = SlotAttentionModel(
    resolution=params.resolution,
    num_slots=params.num_slots,
    num_iterations=params.num_iterations,
    slot_size=params.slot_size,
)
model = SlotAttentionMethod.load_from_checkpoint(CKPT_PATH, model=sa, datamodule=None)
model.eval()

t = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(slightly_off_center_crop),
    ]
)

with h5py.File("data/clevr_with_masks.h5", "r") as f:
    image = f["image"][0]
# image = Image.open("ep000000_t000_fl_c2.png")
# image = image.convert("RGB")
# plt.imshow(image, interpolation='nearest')
# plt.show()
image = t(image)
images = model.predict(
    image,
    do_transforms=True,
    debug=True,
    return_pil=True,
    background_detection="spread_out",
)

plt.imshow(images, interpolation="nearest")
plt.show()
