from transformers import Blip2Processor
from PIL import Image

def preprocess(image_path, question, processor=None):
    if processor is None:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, text=question, return_tensors="pt")
    return inputs
