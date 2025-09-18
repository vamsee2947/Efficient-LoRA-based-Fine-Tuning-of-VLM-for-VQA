from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        quantization_config=bnb_config,
        device_map="auto"
    )

    lora_cfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    # Add your data loading, optimizer, and training loop here
    
if __name__ == "__main__":
    main()
