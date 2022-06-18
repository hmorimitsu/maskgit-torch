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

import os
from pathlib import Path

import requests
from torch import hub


BASE_URL = 'https://github.com/hmorimitsu/maskgit-torch/releases/download/weights/'
CKPT_NAMES = [
    'maskgit_imagenet256.ckpt', 'maskgit_imagenet512.ckpt', 'tokenizer_imagenet256.ckpt', 'tokenizer_imagenet512.ckpt'
]


if __name__ == '__main__':
    save_dir = Path(__file__).parent.absolute() / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)
    for fname in  CKPT_NAMES:
        url = BASE_URL + fname
        save_path = save_dir / fname
        if save_path.exists():
            local_size = os.stat(save_path).st_size
            with requests.get(url, stream=True) as r:
                download_size = int(r.headers.get('Content-Length'))
            if local_size == download_size:
                print(f'{save_path} already exists, skipping download')
                continue
        print(f'Downloading to {url} to {save_path}')
        hub.download_url_to_file(url, save_path, )
