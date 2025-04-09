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

_lang_encoders = {}


def register_lang_encoder(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]

    _lang_encoders[model_name] = fn

    return fn

def lang_encoders(model_name):
    return _lang_encoders[model_name]

def is_lang_encoder(model_name):
    return model_name in _lang_encoders
