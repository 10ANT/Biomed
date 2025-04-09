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


from typing import Tuple
from PIL import ImageDraw, ImageFont, Image
import random


class Model:
    def init(self):
        pass

    def predict(
        self, image: Image.Image, modality_type: str, targets: list[str]
    ) -> Tuple[Image.Image, str]:
        # Randomly split targets into found and not found
        targets_found = random.sample(targets, k=len(targets) // 2)
        targets_not_found = [t for t in targets if t not in targets_found]

        # Create a copy of the image to draw on
        image_with_text = image.copy()
        draw = ImageDraw.Draw(image_with_text)

        # Draw found targets on the image with larger font
        font_size = 36
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            # Fallback to default font if DejaVuSans is not available
            font = ImageFont.load_default()

        # Calculate starting position from bottom
        line_height = 50
        total_height = len(targets_found) * line_height
        padding = 20

        # Start from bottom and work upwards
        y_position = image_with_text.height - total_height - padding
        for target in targets_found:
            draw.text((20, y_position), target, fill="red", font=font)
            y_position += line_height

        # Format targets_not_found as a string with one target per line
        targets_not_found_str = (
            "\n".join(targets_not_found)
            if targets_not_found
            else "All targets were found!"
        )

        return image_with_text, targets_not_found_str
