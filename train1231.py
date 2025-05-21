import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import os
from sklearn.model_selection import train_test_split
import re

def clean_text(text):
    """
    Metni temizler ve normalleştirme işlemleri yapar:
    - Gereksiz boşlukları kaldırır
    - Tüm metni küçük harfe çevirir
    - Türkçe karakterleri İngilizce karakterlere dönüştürür
    """
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir
    text = text.lower()
    
    # Türkçe karakterleri değiştir
    tr_chars = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c'
    }
    for tr_char, en_char in tr_chars.items():
        text = text.replace(tr_char, en_char)
    
    # Gereksiz boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class PhoneDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        # Veriyi temizle ve normalleştir
        self.dataframe = dataframe.copy()
        self.dataframe["input"] = self.dataframe["input"].apply(clean_text)
        self.dataframe["output"] = self.dataframe["output"].apply(clean_text)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        input_text = self.dataframe.iloc[idx]["input"]
        output_text = self.dataframe.iloc[idx]["output"]
        
        # Tokenization - Geliştirilmiş verimlilik için batch işlemi yerine tekli işlem
        input_encoding = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None  # Batch yerine tek örnek dönüşü
        )
        
        output_encoding = self.tokenizer(
            output_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None  # Batch yerine tek örnek dönüşü
        )
        
        # -100 ile padding token'larını maskeleme
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in output_encoding["input_ids"]]
        
        return {
            "input_ids": torch.tensor(input_encoding["input_ids"]),
            "attention_mask": torch.tensor(input_encoding["attention_mask"]),
            "labels": torch.tensor(labels)
        }

def compute_metrics(eval_preds):
    """
    Model değerlendirme metrikleri hesaplama fonksiyonu.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # -100 değerlerini tokenizer'ın pad_token_id'si ile değiştir
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Tahminleri decode et
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Metrikleri hesapla
    exact_match = sum([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_labels)
    
    # Ayrıca token bazlı bir benzerlik metriği ekleyelim (kelime örtüşmesi)
    token_match_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_tokens = set(pred.strip().split())
        label_tokens = set(label.strip().split())
        
        if not label_tokens:  # Boş etiket durumunu kontrol et
            token_match_scores.append(0.0)
            continue
            
        # Kesişim / Birleşim (Jaccard benzerliği)
        intersection = len(pred_tokens.intersection(label_tokens))
        union = len(pred_tokens.union(label_tokens))
        token_match_scores.append(intersection / union if union > 0 else 0.0)
    
    token_similarity = sum(token_match_scores) / len(token_match_scores) if token_match_scores else 0.0
    
    return {
        "exact_match": exact_match,
        "token_similarity": token_similarity
    }

def preprocess_dataset(csv_path, test_size=0.2, val_size=0.1):
    """
    Veri setini yükler, temizler ve eğitim/doğrulama/test kümelerine böler
    """
    # Veri setini oku
    df = pd.read_csv(csv_path)
    
    # Eksik değerleri kontrol et ve temizle
    df = df.dropna()
    
    # Veriyi temizle ve normalleştir
    df["input"] = df["input"].apply(clean_text)
    df["output"] = df["output"].apply(clean_text)
    
    # Veri kümesini böl: önce test kümesini ayır
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Kalan veriyi eğitim ve doğrulama kümelerine böl
    train_size = 1 - (val_size / (1 - test_size))  # Oranı ayarla
    train_df, val_df = train_test_split(train_val_df, train_size=train_size, random_state=42)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Yapılandırma parametreleri
    MODEL_NAME = "google/mt5-small"  # Daha küçük model, daha hızlı eğitim
    MAX_LENGTH = 128
    BATCH_SIZE = 16  # Daha büyük batch size
    EPOCHS = 5
    LEARNING_RATE = 3e-4  # Adam optimizer için önerilen değer
    WEIGHT_DECAY = 0.01
    
    # Veri setini hazırla
    train_df, val_df, test_df = preprocess_dataset("telefon_dataset.csv")
    
    print(f"Eğitim veri seti boyutu: {len(train_df)}")
    print(f"Doğrulama veri seti boyutu: {len(val_df)}")
    print(f"Test veri seti boyutu: {len(test_df)}")
    
    # Tokenizer ve model yükleme
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # GPU kullanımını kontrol et
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    model.to(device)
    
    # Veri setlerini oluştur
    train_dataset = PhoneDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_dataset = PhoneDataset(val_df, tokenizer, max_length=MAX_LENGTH)
    test_dataset = PhoneDataset(test_df, tokenizer, max_length=MAX_LENGTH)
    
    # Eğitim argümanlarını yapılandır
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),  # FP16 ile eğitimi hızlandır
        gradient_accumulation_steps=2,  # Bellek tasarrufu için gradient biriktirme
        load_best_model_at_end=True,
        metric_for_best_model="token_similarity"
    )
    
    # Veri toplayıcı tanımla
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest"  # Batch içindeki en uzun örneğe göre padding yap (bellek verimliliği)
    )
    
    # Trainer oluştur
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Modeli eğit
    print("Model eğitimi başlatılıyor...")
    trainer.train()
    
    # Test veri setiyle değerlendir
    print("Test veri seti üzerinde değerlendirme yapılıyor...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test sonuçları: {test_results}")
    
    # Modeli kaydet
    model_path = "./phone_query_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model ve tokenizer kaydedildi: {model_path}")
    
    # Örnek tahmin için basit bir fonksiyon
    def predict_query(input_text):
        cleaned_input = clean_text(input_text)
        inputs = tokenizer(cleaned_input, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=50)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction
    
    # Birkaç örnek üzerinde test et
    test_inputs = [
        "8 gb ram ve 128 gb hafızalı bir telefon istiyorum",
        "3000 tl altında bir android telefon öner"
    ]
    
    print("\nÖrnek tahminler:")
    for test_input in test_inputs:
        prediction = predict_query(test_input)
        print(f"Girdi: {test_input}")
        print(f"Tahmin: {prediction}")
        print("-" * 50)
