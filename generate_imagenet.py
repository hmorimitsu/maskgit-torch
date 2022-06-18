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
import math
from pathlib import Path

from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm

from maskgit.inference import ImageNet_class_conditional_generator


def _init_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('-k', '--samples_per_class', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-s', '--image_size', type=int, choices=(256, 512), default=256)
    parser.add_argument('-c', '--classes', type=str, default='')
    parser.add_argument('-o', '--output_dir', type=str, default='output_generation')
    return parser


def main(args: Namespace):
    generator = ImageNet_class_conditional_generator(image_size=args.image_size)
    generator.maskgit_cf.eval_batch_size = args.batch_size
    if len(args.classes) == 0:
        classes = [i for i in range(1000)]
    else:
        if '-' in args.classes:
            start, end = args.classes.split('-')
            classes = [i for i in range(int(start), int(end))]
        else:
            classes = [int(x) for x in args.classes.split(',')]
    
    for icls in tqdm(classes):
        output_dir = Path(args.output_dir) / f'{icls:04d}'
        output_dir.mkdir(parents=True, exist_ok=True)

        input_tokens = generator.create_input_tokens_normal(icls)
        num_iters = math.ceil(float(args.samples_per_class) / args.batch_size)
        for i in range(num_iters):
            inputs = input_tokens.clone()
            if i == (num_iters - 1):
                mod = args.samples_per_class % args.batch_size
                if mod > 0:
                    inputs = inputs[:mod]
            results = generator.generate_samples(inputs)
            results = rearrange(results.detach().cpu().numpy(), 'b c h w -> b h w c')
            images = (np.clip(results, 0, 1) * 255).astype(np.uint8)

            for j, img in enumerate(images):
                img = Image.fromarray(img)
                img.save(output_dir / f'{(i*args.batch_size+j):04d}.png')


if __name__ == '__main__':
    parser: ArgumentParser = _init_parser()
    args: Namespace = parser.parse_args()
    main(args)
