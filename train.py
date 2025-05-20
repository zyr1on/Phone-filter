from sklearn.model_selection import train_test_split
import pandas as pd
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to create training data
def create_training_data():
    """Örneklem eğitim verileri oluşturur ve ön işleme uygular."""
    
    training_data = []

    # Ön işleme fonksiyonu
    def preprocess_text(text):
        # 1. Tüm metni küçük harfe dönüştür
        text = text.lower()
        
        # 2. Sayı olmayan ve harf olmayan karakterleri boşlukla değiştir
        # (Örn: !, ?, ., ,, vb. kaldırır, sayıları korur)
        # Sadece harf, sayı ve boşluk bırakmak istiyorsak:
        # text = re.sub(r'[^a-zA-Z0-9çÇğĞıİöÖşŞüÜ\s]', '', text)
        # Ancak, bu örnekte 'TL' veya 'GB' gibi ifadelerin bitişik kalması daha iyi olabilir.
        # Bu nedenle, sadece noktalama işaretlerini hedefleyelim:
        text = re.sub(r'[^\w\s]', '', text) # Alfabetik karakterler, sayılar ve boşluk dışındakileri kaldırır
        
        # Fazla boşlukları tek boşluğa dönüştür
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Veri örnekleri (eski haliyle aynı, preprocess_text ile işlenecek)
    # ... (buraya orijinal price_examples, ram_examples, os_examples vb. eklenecek)
    # Kopyala yapıştır kolaylığı için kısaltılmış örnekler:
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
        ("Ucuz bir telefon bakıyorum", "price ASC"),
        ("Pahalı telefonlar ilgimi çekiyor", "price DESC"),
        ("18000 TL'ye kadar bir model arıyorum", "price <= 18000"),
        ("Fiyatı 9000 TL'yi geçmesin", "price < 9000"),
        ("3000 TL ile 6000 TL arasında telefonlar", "price >= 3000 AND price <= 6000"),
        ("Bütçe dostu telefonlar", "price ASC"),
    ]
    
    # Enhanced RAM Examples
    ram_examples = [
        ("RAM'i 6GB'tan fazla olsun", "ram > 6"),
        ("En az 8GB RAM", "ram >= 8"),
        ("RAM 12GB olsun", "ram = 12"),
        ("16GB RAM olan telefonlar", "ram = 16"),
        ("RAM'i 6-8 GB arası", "ram >= 6 AND ram <= 8"),
        ("RAM'i yüksek telefonlar", "ram DESC"),
        ("6 GB RAM'den az olmayan telefonlar", "ram >= 6"),
        ("Minimum 4GB RAM", "ram >= 4"),
        ("Çok hızlı, yüksek RAM'li bir telefon", "ram DESC"),
        ("8 gigabayt RAM'li telefonlar", "ram = 8"),
    ]
    
    # Enhanced Operating System/Brand Examples
    os_examples = [
        ("Android telefonlar", "os = 'Android'"),
        ("iOS işletim sistemli telefonlar", "os = 'iOS'"),
        ("iPhone önerisi istiyorum", "os = 'iOS'"),
        ("Samsung telefonlar", "brand = 'Samsung'"),
        ("Sadece Apple telefon olsun", "brand = 'Apple'"),
        ("Xiaomi telefonları göster", "brand = 'Xiaomi'"),
        ("Huawei markalı telefonlar", "brand = 'Huawei'"),
        ("Google Pixel telefonlar", "brand = 'Google'"),
        ("En son çıkan Android'ler", "os = 'Android' AND release_date DESC"),
        ("En iyi iOS deneyimi", "os = 'iOS'"),
    ]
    
    # Enhanced Battery Examples
    battery_examples = [
        ("Bataryası güçlü telefonlar", "battery DESC"),
        ("5000 mAh üzeri batarya", "battery > 5000"),
        ("Bataryası en az 4000 mAh olsun", "battery >= 4000"),
        ("Batarya kapasitesi yüksek telefonlar", "battery DESC"),
        ("Uzun pil ömrü olan telefonlar", "battery DESC"),
        ("6000 mAh bataryalı telefonlar", "battery = 6000"),
        ("Düşük batarya tüketimi olan telefonlar", "battery ASC"), # Might imply smaller battery, or efficiency, tricky
        ("Bataryası 4500 mAh ve üstü", "battery >= 4500"),
    ]
    
    # Enhanced Camera Examples
    camera_examples = [
        ("Kamerası iyi telefonlar", "camera DESC"),
        ("En iyi kameralı telefonlar", "camera DESC"),
        ("48 MP üzeri kamera", "camera > 48"),
        ("Kamera çözünürlüğü yüksek olanlar", "camera DESC"),
        ("Kamerası en az 64 MP olsun", "camera >= 64"),
        ("Ön kamerası başarılı telefonlar", "front_camera DESC"),
        ("Çift kameralı telefonlar", "num_cameras = 2"), # Example, assumes this feature exists
        ("108 MP ana kameralı telefon", "camera = 108"),
        ("Gece modu iyi olan telefonlar", "camera DESC"), # Implies camera quality
    ]

    # Enhanced Screen Examples
    screen_examples = [
        ("Büyük ekranlı telefonlar", "screen_size DESC"),
        ("6.5 inç üzeri ekranlı telefonlar", "screen_size > 6.5"),
        ("Kompakt boyutlu telefonlar", "screen_size ASC"),
        ("AMOLED ekranlı telefonlar", "screen_type = 'AMOLED'"),
        ("Yüksek yenileme hızlı ekranlar", "refresh_rate DESC"),
        ("Full HD+ ekranlı telefonlar", "resolution = 'FHD+'"), # Example
    ]

    # Enhanced Storage Examples
    storage_examples = [
        ("256 GB depolama alanı", "storage = 256"),
        ("512 GB hafızalı telefonlar", "storage = 512"),
        ("Yeterli depolama alanı olan telefonlar", "storage DESC"),
        ("128 GB'tan az olmasın", "storage >= 128"),
    ]

    # Enhanced Mixed Examples
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
        ("6.5 inç üzeri ekranlı, bataryası iyi telefonlar", "screen_size > 6.5 AND battery DESC"),
        ("Hem ucuz hem de iyi kameralı telefonlar", "price ASC AND camera DESC"),
        ("Samsung veya Apple, RAM'i 8GB'tan fazla", "brand IN ('Samsung', 'Apple') AND ram > 8"),
        ("Hızlı işlemcili Android telefonlar", "os = 'Android' AND processor DESC"),
        ("256 GB depolama ve 6.2 inç ekranlı", "storage = 256 AND screen_size = 6.2"),
        ("En yeni model Xiaomi telefonları", "brand = 'Xiaomi' AND release_date DESC"),
        ("Suya dayanıklı ve bataryası güçlü telefonlar", "water_resistant = TRUE AND battery DESC"),
        ("Oyun için yüksek performanslı telefon", "processor DESC AND ram DESC AND refresh_rate DESC"),
        ("Bütçem 12000 TL, Samsung marka ve kamerası iyi olsun", "price <= 12000 AND brand = 'Samsung' AND camera DESC"),
        ("Android 13 işletim sistemli telefonlar", "os_version = 'Android 13'"), # Example
        ("Kamerası 64 MP ve fiyatı 15000 TL altı", "camera = 64 AND price < 15000"),
        ("RAM'i 8 GB ve depolaması 128 GB olan Android telefonlar", "ram = 8 AND storage = 128 AND os = 'Android'"),
        ("En uygun fiyatlı iPhone'lar", "brand = 'Apple' AND price ASC"),
        ("4000 mAh üzeri batarya ve 6GB RAM", "battery > 4000 AND ram = 6"),
        ("Yüksek ekran yenileme hızı ve güçlü işlemci", "refresh_rate DESC AND processor DESC"),
    ]
    
    # Tüm örnekleri birleştir
    all_examples = price_examples + ram_examples + os_examples + battery_examples + camera_examples + screen_examples + storage_examples + mixed_examples
    
    # Veri setini oluştur ve preprocess_text fonksiyonunu uygula
    for prompt, filter_query in all_examples:
        # Sadece prompt'u ön işleme alıyoruz, filter_query'yi olduğu gibi bırakıyoruz
        # çünkü filter_query SQL benzeri bir yapıda ve ona dokunmak istemeyiz.
        processed_prompt = preprocess_text(prompt)
        training_data.append({
            "prompt": processed_prompt,
            "filter_query": filter_query
        })
    
    return training_data

def main():
    # Create the dataset
    print("Veri seti oluşturuluyor...")
    data = create_training_data()

    # Split the dataset into training and evaluation sets
    train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

    # Convert to HuggingFace Datasets format
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

    # Use mT5 model (multilingual and for seq2seq)
    model_name = "google/mt5-small"
    print(f"Model yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) # Move model to device

    # Data preparation function
    def preprocess_function(examples):
        inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=128)
        outputs = tokenizer(examples["filter_query"], truncation=True, padding="max_length", max_length=128)
        
        # Create decoder input_ids for labels
        batch = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": outputs.input_ids, # labels directly from tokenized output
        }
        
        # Replace padding token id in labels with -100 so it's ignored in loss calculation
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] 
            for labels in batch["labels"]
        ]
        
        return batch

    # Tokenize the datasets
    print("Veri setleri tokenize ediliyor...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "filter_query"])
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "filter_query"])

    # Define training arguments
    from transformers.trainer_utils import IntervalStrategy
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./phone_filter_model",
        learning_rate=3e-5, # Slightly reduced learning rate
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, 
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=30, # Increased epochs
        predict_with_generate=True,
        logging_dir="./logs",
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none", # Ensure this is set if not using external logging tools
        fp16=torch.cuda.is_available(), # Use mixed precision training if CUDA is available
    )

    # Create the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    print("🚀 Eğitim başlatılıyor...")
    trainer.train()
    print("✅ Eğitim tamamlandı.")

    # Save the trained model
    model_save_path = "./phone_filter_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"📦 Model kaydedildi: {model_save_path}")

    # Test the model
    def generate_filter_query(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        predicted_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return predicted_query

    # Test with a few examples
    test_prompts = [
        "Fiyatı 10000 TL altı ve 8 GB RAM olan telefonlar",
        "Android telefonlar bataryası yüksek olanlar",
        "En iyi kameralı Samsung telefonlar",
    ]

    print("\nTest sonuçları:")
    for prompt in test_prompts:
        predicted_query = generate_filter_query(prompt)
        print(f"Prompt: {prompt}")
        print(f"Tahmin edilen filtre sorgusu: {predicted_query}\n")

    # Save all training data to a JSON file
    json_path = "training_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Eğitim verileri kaydedildi: {json_path}")

if __name__ == "__main__":
    main()
