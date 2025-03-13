import base64

import litellm

IMAGE_SAMPLE_PATH = (
    "/Users/strickvl/coding/hinbox/discovery_do_not_commit/discovery-sample1.png"
)

# load the image as base64 string
with open(IMAGE_SAMPLE_PATH, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")


response = litellm.completion(
    model="ollama/llava",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Whats in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                    },
                },
            ],
        }
    ],
)
print(response)
