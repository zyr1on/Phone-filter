import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import torch
from datasets import Dataset
import re

# Veri setini oluşturacağız
def create_training_data():
    """Örneklem eğitim verileri oluşturur."""
    
    training_data = []
    
    # Fiyat örnekleri
    price_examples = [
        ("Fiyatı 10000'den az olsun", "price < 10000"),
        ("10 bin tl altı telefonlar", "price < 10000"),
        ("Fiyatı 10000 TL altında olan telefonlar", "price < 10000"),
        ("15000 TL'den pahalı olmasın lütfen", "price < 15000"),
        ("15000 TL fiyat sınırım var", "price < 15000"),
        ("5000 TL civarı telefonlar", "price ~ 5000"),
        ("Fiyatı 8000-12000 TL arası olsun", "price >= 8000 AND price <= 12000"),
        ("Bütçem 20000 TL", "price <= 20000"),
        ("En fazla 12000 TL'ye kadar", "price <= 12000"),
        ("En ucuz telefonlar", "price ASC"),
        ("En pahalı telefonlar", "price DESC"),
        ("20000 TL'den daha pahalı telefonlar", "price > 20000"),
        ("Fiyatı 7000 ile 14000 arasında", "price >= 7000 AND price <= 14000"),
    ]
    
    # RAM örnekleri
    ram_examples = [
        ("RAM'i 6GB'tan fazla olsun", "ram > 6"),
        ("En az 8GB RAM", "ram >= 8"),
        ("RAM 12GB olsun", "ram = 12"),
        ("16GB RAM olan telefonlar", "ram = 16"),
        ("RAM'i 6-8 GB arası", "ram >= 6 AND ram <= 8"),
        ("RAM'i yüksek telefonlar", "ram DESC"),
        ("6 GB RAM'den az olmayan telefonlar", "ram >= 6"),
    ]
    
    # İşletim sistemi örnekleri
    os_examples = [
        ("Android telefonlar", "os = 'Android'"),
        ("iOS işletim sistemli telefonlar", "os = 'iOS'"),
        ("iPhone önerisi istiyorum", "os = 'iOS'"),
        ("Samsung telefonlar", "brand = 'Samsung'"),
        ("Sadece Apple telefon olsun", "brand = 'Apple'"),
        ("Xiaomi telefonları göster", "brand = 'Xiaomi'"),
    ]
    
    # Batarya örnekleri
    battery_examples = [
        ("Bataryası güçlü telefonlar", "battery DESC"),
        ("5000 mAh üzeri batarya", "battery > 5000"),
        ("Bataryası en az 4000 mAh olsun", "battery >= 4000"),
        ("Batarya kapasitesi yüksek telefonlar", "battery DESC"),
    ]
    
    # Kamera örnekleri
    camera_examples = [
        ("Kamerası iyi telefonlar", "camera DESC"),
        ("En iyi kameralı telefonlar", "camera DESC"),
        ("48 MP üzeri kamera", "camera > 48"),
        ("Kamera çözünürlüğü yüksek olanlar", "camera DESC"),
        ("Kamerası en az 64 MP olsun", "camera >= 64"),
    ]
    
    # Karışık örnekler
    mixed_examples = [
        ("Fiyatı 10000den az, RAMi 6dan yukarı telefonlar", "price < 10000 AND ram > 6"),
        ("8GB RAM ve 5000 mAh bataryalı telefonlar", "ram = 8 AND battery >= 5000"),
        ("Android, 12GB RAM ve fiyatı 15000 TL altı", "os = 'Android' AND ram = 12 AND price < 15000"),
        ("Xiaomi marka, bataryası 4500 mAh üzeri ve fiyatı 12000 TL altı", "brand = 'Xiaomi' AND battery > 4500 AND price < 12000"),
        ("iPhone, 256 GB depolama ve kamerası iyi olan", "os = 'iOS' AND storage = 256 AND camera DESC"),
        ("Fiyatı 8000-15000 TL arası, Android ve en az 8GB RAM", "price >= 8000 AND price <= 15000 AND os = 'Android' AND ram >= 8"),
        ("Samsung marka, 20000 TL altı, RAM'i yüksek", "brand = 'Samsung' AND price < 20000 AND ram DESC"),
        ("Bataryası güçlü ve 12000 TL altı telefonlar", "battery DESC AND price < 12000"),
        ("En iyi kameralı Android telefonlar", "camera DESC AND os = 'Android'"),
        ("iyi bir oyun telefonu istiyorum", "ram DESC AND processor DESC"),
        ("Uygun fiyatlı iyi bir telefon arıyorum", "price ASC AND (ram >= 6 OR camera >= 48)"),
        ("10000 TL altı iyi kameralı telefonlar", "price < 10000 AND camera DESC"),
        ("Bataryası uzun ömürlü ve Android telefonlar", "battery DESC AND os = 'Android'"),
        ("En yeni Apple telefonları göster", "brand = 'Apple' AND release_date DESC"),
        ("6.5 inç üzeri ekranlı telefonlar", "screen_size > 6.5"),
    ]
    
    # Tüm örnekleri birleştir
    all_examples = price_examples + ram_examples + os_examples + battery_examples + camera_examples + mixed_examples
    
    # Veri setini oluştur
    for prompt, filter_query in all_examples:
        training_data.append({
            "prompt": prompt,
            "filter_query": filter_query
        })
    
    return training_data

# Veri setini oluşturalım
data = create_training_data()

# Veri setini eğitim ve değerlendirme olarak bölelim
train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

# HuggingFace Datasets formatına dönüştürelim
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

# Model ve tokenizer yükleyelim (Türkçe için dbmdz/bert-base-turkish-cased kullanacağız)
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Modelimizi sequence-to-sequence olarak değil, text-generation olarak kullanacağız
# Burada BERT sınıflandırma değil text generation yapabilen bir model kullanacağız
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Veri hazırlama fonksiyonu
def preprocess_function(examples):
    return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=128)

# Veri setlerini tokenize edelim
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# Özel trainer tanımlama (sequence generation için)
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM

# Daha iyi bir seçenek olarak mT5 modelini kullanalım (çok dilli ve seq2seq için)
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Veri hazırlama fonksiyonunu güncelleyelim
def preprocess_function(examples):
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=128)
    outputs = tokenizer(examples["filter_query"], truncation=True, padding="max_length", max_length=128)
    
    # Decoder input_ids oluştur
    batch = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": outputs.input_ids.copy(),
    }
    
    # -100 değerini padding token_id'leri için kullan (loss hesaplamasında göz ardı edilecek)
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] 
        for labels in batch["labels"]
    ]
    
    return batch

# Veri setlerini güncellenmiş şekilde tokenize edelim
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# Eğitim argümanlarını tanımlayalım
training_args = Seq2SeqTrainingArguments(
    output_dir="./phone_filter_model",
    
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=50,
    predict_with_generate=True,
    logging_dir="./logs",
    warmup_steps=500,
    save_strategy="epoch",
    gradient_accumulation_steps=2
)

# Trainer'ı oluşturalım
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
)

# Modeli eğitelim
print("Model eğitimine başlanıyor...")
trainer.train()

# Eğitilen modeli kaydedelim
trainer.save_model("./phone_filter_model")
tokenizer.save_pretrained("./phone_filter_model")

print("Model eğitimi tamamlandı ve model kaydedildi!")

# Test edelim
def generate_filter_query(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    predicted_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_query

# Birkaç örnek test edelim
test_prompts = [
    "Fiyatı 10000 TL altı ve 8 GB RAM olan telefonlar",
    "Android telefonlar bataryası yüksek olanlar",
    "En iyi kameralı Samsung telefonlar"
]

print("\nTest sonuçları:")
for prompt in test_prompts:
    predicted_query = generate_filter_query(prompt)
    print(f"Prompt: {prompt}")
    print(f"Tahmin edilen filtre sorgusu: {predicted_query}\n")

# Tüm eğitim verilerini JSON dosyasına kaydedelim
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Eğitim verileri training_data.json dosyasına kaydedildi.")