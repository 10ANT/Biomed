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

from fvcore.common.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    2. Support config versioning.
       When attempting to merge an old config, it will convert the old config automatically.

    .. automethod:: clone
    .. automethod:: freeze
    .. automethod:: defrost
    .. automethod:: is_frozen
    .. automethod:: load_yaml_with_base
    .. automethod:: merge_from_list
    .. automethod:: merge_from_other_cfg
    """

    def merge_from_dict(self, dict):
        pass
    
node = CfgNode()