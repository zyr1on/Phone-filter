import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

# Güvenli token temizleme
def sanitize(token_list, tokenizer):
    return [
        [int(t) for t in seq if isinstance(t, int) and 0 <= t <= tokenizer.vocab_size]
        for seq in token_list
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="saved_model", help="Modelin kaydedildiği klasör")
    args = parser.parse_args()

    # Model ve tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.eval()

    # Test verisi - örnek girişler buraya yazılır
    input_texts = [
        "6 gb üzeri android telefon öner",
        "12 bin TL altı iPhone öner",
        "Geniş ekranlı, ucuz Samsung telefon"
    ]

    # Girişleri tokenize et
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(model.device)

    # Tahminleri al
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64)

    # Tensor -> List[List[int]]
    preds = outputs.tolist()
    preds = sanitize(preds, tokenizer)

    # Decode et
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = [s if s else "" for s in decoded_preds]

    # Sonuçları yazdır
    for inp, pred in zip(input_texts, decoded_preds):
        print(f"\nGirdi: {inp}\nÇıktı: {pred}")

if __name__ == "__main__":
    main()
