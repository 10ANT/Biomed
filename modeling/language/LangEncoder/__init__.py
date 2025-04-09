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

from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer

from .transformer import *
from .build import *


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']

    if not is_lang_encoder(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return lang_encoders(model_name)(config_encoder, tokenizer, verbose, **kwargs)

def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if config_encoder['TOKENIZER'] == 'clip':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    elif config_encoder['TOKENIZER'] == 'clip-fast':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer, from_slow=True)
    elif config_encoder['TOKENIZER'] == 'biomed-clip':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_encoder['TOKENIZER'])

    return tokenizer