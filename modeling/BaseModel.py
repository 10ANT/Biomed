# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

import torch
import torch.nn as nn

from utilities.model import align_and_update_state_dicts

from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files

import huggingface_hub

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def save_pretrained(self, save_dir):
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model_state_dict.pt"))

    def from_pretrained(self, pretrained, filename: str = "biomedparse_v1.pt",
                        local_dir: str = "./pretrained", config_dir: str = "./configs"):
        if pretrained.startswith("hf_hub:"):
            hub_name = pretrained.split(":")[1]
            huggingface_hub.hf_hub_download(hub_name, filename=filename, 
                                            local_dir=local_dir)
            huggingface_hub.hf_hub_download(hub_name, filename="config.yaml", 
                                            local_dir=config_dir)
            load_dir = os.path.join(local_dir, filename)
        else:
            load_dir = pretrained
        
        state_dict = torch.load(load_dir, map_location=self.opt['device'])
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self