import base64

import torch
from rich import print
from transformers import AutoModelForImageTextToText, AutoProcessor

IMAGE_SAMPLE_PATH = (
    "/home/strickvl/coding/hinbox/discovery_do_not_commit/discovery-sample1.png"
)
ARABIC_IMAGE_SAMPLE_PATH = (
    "/home/strickvl/coding/hinbox/discovery_do_not_commit/arabic-sample2.png"
)

# load the image as base64 string
with open(ARABIC_IMAGE_SAMPLE_PATH, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")


# response = litellm.completion(
#     model="ollama/llama3.2-vision",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Whats in this image?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": base64_image,
#                     },
#                 },
#             ],
#         }
#     ],
# )
# print(response)

# prompt = "This is in Arabic. Translate what you see into English."

# messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

# # model = "ollama/llama3.2-vision"
# model = "ollama/gemma3:27b"

# response = completion(
#     model=model,
#     messages=messages,
#     api_base="http://192.168.178.175:11434",
#     images=[
#         f"data:image/jpeg;base64,{base64_image}"
#     ],  # <---- if this is supplied with encoded images list it will be properly added
# )

# print(response)

model_id = "CohereForAI/aya-vision-8b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16
)

# Format message with the aya-vision chat template
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "This is in Arabic. Translate what you see into English.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    padding=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

gen_tokens = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.1,
)

print(
    processor.tokenizer.decode(
        gen_tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
)
