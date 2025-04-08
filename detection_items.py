import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from transformers import YolosForObjectDetection

# Load the model config
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-base")

# Get all detectable object names
object_names = list(model.config.id2label.values())

# Display the object names
for name in object_names:
    print(name)
