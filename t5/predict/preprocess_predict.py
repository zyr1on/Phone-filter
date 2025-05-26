import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import unicodedata

def normalize_text(text):
    """Küçük harfe çevirir ve Türkçe karakterleri normalize eder (ç -> c, ı -> i, vs.)."""
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

class PhoneT5Predictor:
    def __init__(self, model_path='./phone_t5_model'):
        print("T5 Model yükleniyor...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model klasörü bulunamadı: {model_path}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model başarıyla yüklendi! Cihaz: {self.device}")
    
    def predict(self, input_text, max_length=256, temperature=0.7):
        input_text = normalize_text(input_text)
        
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                temperature=temperature,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def predict_batch(self, input_texts, max_length=256):
        normalized_inputs = [normalize_text(text) for text in input_texts]
        
        inputs = self.tokenizer(
            normalized_inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        responses = []
        for output in outputs:
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses

def main():
    try:
        predictor = PhoneT5Predictor()
        
        print("\n" + "="*60)
        print("T5 Telefon Öneri Sistemi")
        print("Çıkmak için 'quit' yazın")
        print("Toplu test için 'test' yazın")
        print("="*60)
        
        while True:
            user_input = input("\nTelefon isteğinizi yazın: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'çık', 'çıkış']:
                print("Görüşürüz!")
                break
            
            if user_input.lower() == 'test':
                test_inputs = [
                    "15000 tl den az 6 gb ram android telefon öner",
                    "fiyatı 10000 tl olan 4 gb iphone telefon öner",
                    "25000 tl hafızası iyi olan oyun için telefon android",
                    "fotoğraf çekmek için samsung telefon",
                    "8 gb ram oyun telefonu"
                ]
                
                print("\nToplu test yapılıyor...")
                results = predictor.predict_batch(test_inputs)
                
                for i, (inp, out) in enumerate(zip(test_inputs, results)):
                    print(f"\nTest {i+1}:")
                    print(f"Girdi: {inp}")
                    print(f"Çıktı: {out}")
                continue
            
            if not user_input:
                print("Lütfen bir istek yazın.")
                continue
            
            print("Tahmin yapılıyor...")
            
            try:
                result = predictor.predict(user_input)
                print(f"\nSonuç: {result}")
                
                print("\nFarklı yaratıcılık seviyeleri:")
                for temp in [0.3, 0.7, 1.0]:
                    alt_result = predictor.predict(user_input, temperature=temp)
                    print(f"  T={temp}: {alt_result}")
                    
            except Exception as e:
                print(f"Tahmin hatası: {e}")
    
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        print("Önce train.py'yi çalıştırarak modeli eğitin!")
    
    except Exception as e:
        print(f"Genel hata: {e}")
        print("Gerekli paketler: pip install torch transformers")

if __name__ == "__main__":
    main()
