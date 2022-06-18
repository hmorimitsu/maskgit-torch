# Copyright 2022 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm

from maskgit.configs import vqgan_config
from maskgit.nets.vqgan_tokenizer import VQVAE
from maskgit.inference import ImageNet_class_conditional_generator

class ImageDataset(Dataset):
    def __init__(
        self,
        imagenet_images_dir: Union[str, Path],
        target_size: int = 256,
        images_range: Optional[Tuple[int, int]] = (0, -1)
    ) -> None:
        super().__init__()
        self.target_size = target_size

        imagenet_images_dir = Path(imagenet_images_dir)
        self.image_paths = sorted([p for p in imagenet_images_dir.glob('**/*') if self._is_image_path(p)])
        if images_range[1] > 0:
            self.image_paths = self.image_paths[images_range[0]:images_range[1]]
        print(f'Found {len(self.image_paths)} images to reconstruct')

        self.transform = torchvision.transforms.ToTensor()
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index])
        image = image.convert('RGB')
        image = self._resize_and_crop(image)
        image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.image_paths)

    def _is_image_path(
        self,
        path: Path
    ) -> bool:
        ext = path.name[path.name.rfind('.')+1:]
        if (path.is_dir()
                or path.name.startswith('.')
                or len(ext) == 0
                or ext.lower() not in ('jpg', 'jpeg', 'png')):
            return False
        return True

    def _resize_and_crop(
        self,
        image: Image.Image
    ) -> Image.Image:
        w, h = image.size
        min_size = min(w, h)
        scale = float(self.target_size) / min_size
        rw, rh = int(scale*w), int(scale*h)
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
        top = (rh - self.target_size) // 2
        bottom = top + self.target_size
        left = (rw - self.target_size) // 2
        right = left + self.target_size
        image = image.crop((left, top, right, bottom))
        return image


def _init_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-i', '--images_dir', type=str, required=True, help='Path to the root directory where the images are')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Number of images in a minibatch')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of worker threads to load the images')
    parser.add_argument('-s', '--image_size', type=int, choices=(256, 512), default=256, help='Size of the reconstructed image')
    parser.add_argument('-o', '--output_dir', type=str, default='output_reconstruction', help='Path to a directory where the outputs will be saved')
    parser.add_argument('-w', '--write_input_image', action='store_true', help='If set, the input images will also be saved to the output directory')
    parser.add_argument('-r', '--images_range', type=int, nargs=2, default=(0, -1),
                        help=('Optional. To use provide two values indicating the starting and ending indices of the images to be loaded.'
                              ' Only images within this range will be loaded. Useful for manually sharding the dataset if running all images'
                              ' at once would take too much time.'))
    return parser


def main(args: Namespace) -> None:
    dataset = ImageDataset(args.images_dir, args.image_size, args.images_range)
    dataloader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)

    config = vqgan_config.get_config()
    model = VQVAE(config)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    ckpt = torch.load(ImageNet_class_conditional_generator.checkpoint_canonical_path("tokenizer", args.image_size))
    model.load_state_dict(ckpt['state_dict'])

    if args.write_input_image:
        in_output_dir = Path(args.output_dir) / 'inputs'
        in_output_dir.mkdir(parents=True, exist_ok=True)
    rec_output_dir = Path(args.output_dir) / 'reconstructions'
    rec_output_dir.mkdir(parents=True, exist_ok=True)

    for i, x in enumerate(tqdm(dataloader)):
        if torch.cuda.is_available():
            x = x.cuda()
        
        if args.write_input_image:
            write_images(x, in_output_dir, i, args.batch_size, args.images_range[0])

        input_dict = {'image': x}
        xrec = model(input_dict)
        xrec = xrec.clamp(0, 1)
        
        write_images(xrec, rec_output_dir, i, args.batch_size, args.images_range[0])


def write_images(
    images: torch.Tensor,
    output_dir: Path,
    i: int,
    batch_size: int,
    start_offset: int
) -> None:
    images = (255 * images.permute(0, 2, 3, 1).detach().cpu().numpy()).astype(np.uint8)
    for j, img in enumerate(images):
        k = i * batch_size + j + start_offset
        img = Image.fromarray(img)
        img.save(output_dir / f'{k:08d}.png')


if __name__ == '__main__':
    parser: ArgumentParser = _init_parser()
    args: Namespace = parser.parse_args()
    main(args)
