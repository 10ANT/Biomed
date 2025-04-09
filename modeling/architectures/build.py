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

_model_entrypoints = {}


def build_model(config, **kwargs):
    model_name = config['MODEL']['NAME']

    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, **kwargs)

def register_model(fn):
    module_name_split = fn.__module__.split('.')
    model_name = module_name_split[-1]
    _model_entrypoints[model_name] = fn
    return fn

def model_entrypoints(model_name):
    return _model_entrypoints[model_name]

def is_model(model_name):
    return model_name in _model_entrypoints