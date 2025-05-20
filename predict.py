import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Modeli ve tokenizer'ı yükle
model_path = "./phone_filter_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def predict_filter_query(prompt):
    """
    Verilen doğal dil prompt'unu filtre sorgusuna dönüştürür.
    
    Args:
        prompt: Kullanıcının doğal dil isteği
        
    Returns:
        filter_query: SQL benzeri filtre sorgusu
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
    # Modelden tahmin alalım
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    # Tahmin edilen filtre sorgusunu çözelim
    predicted_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return predicted_query

def filter_phones_with_query(csv_file, filter_query):
    """
    CSV dosyasını verilen filtre sorgusuna göre filtreler.
    
    Args:
        csv_file: Telefon verilerini içeren CSV dosyası yolu
        filter_query: SQL benzeri filtre sorgusu
        
    Returns:
        filtered_df: Filtrelenmiş pandas DataFrame
    """
    # CSV dosyasını oku
    df = pd.read_csv(csv_file)
    
    # Filtre sorgusunu işle
    # Bu basit bir örnek. Gerçek uygulamada güvenlik önlemleri alınmalıdır.
    # Filtre sorgusu bileşenlerine ayır
    query_parts = filter_query.split("AND")
    
    # Her parçayı ayrı ayrı değerlendir
    result_df = df.copy()
    
    for part in query_parts:
        part = part.strip()
        
        # Fiyat filtreleri
        if "price <" in part:
            price_limit = int(part.split("<")[1].strip())
            result_df = result_df[result_df["price"] < price_limit]
        elif "price <=" in part:
            price_limit = int(part.split("<=")[1].strip())
            result_df = result_df[result_df["price"] <= price_limit]
        elif "price >" in part:
            price_limit = int(part.split(">")[1].strip())
            result_df = result_df[result_df["price"] > price_limit]
        elif "price >=" in part:
            price_limit = int(part.split(">=")[1].strip())
            result_df = result_df[result_df["price"] >= price_limit]
        elif "price =" in part:
            price_val = int(part.split("=")[1].strip())
            result_df = result_df[result_df["price"] == price_val]
        elif "price ASC" in part:
            result_df = result_df.sort_values(by="price", ascending=True)
        elif "price DESC" in part:
            result_df = result_df.sort_values(by="price", ascending=False)
        
        # RAM filtreleri
        elif "ram <" in part:
            ram_limit = int(part.split("<")[1].strip())
            result_df = result_df[result_df["ram"] < ram_limit]
        elif "ram <=" in part:
            ram_limit = int(part.split("<=")[1].strip())
            result_df = result_df[result_df["ram"] <= ram_limit]
        elif "ram >" in part:
            ram_limit = int(part.split(">")[1].strip())
            result_df = result_df[result_df["ram"] > ram_limit]
        elif "ram >=" in part:
            ram_limit = int(part.split(">=")[1].strip())
            result_df = result_df[result_df["ram"] >= ram_limit]
        elif "ram =" in part:
            ram_val = int(part.split("=")[1].strip())
            result_df = result_df[result_df["ram"] == ram_val]
        elif "ram DESC" in part:
            result_df = result_df.sort_values(by="ram", ascending=False)
        
        # İşletim sistemi filtreleri
        elif "os =" in part:
            os_val = part.split("=")[1].strip().strip("'")
            result_df = result_df[result_df["os"] == os_val]
        
        # Marka filtreleri
        elif "brand =" in part:
            brand_val = part.split("=")[1].strip().strip("'")
            result_df = result_df[result_df["brand"] == brand_val]
        
        # Batarya filtreleri
        elif "battery <" in part:
            battery_limit = int(part.split("<")[1].strip())
            result_df = result_df[result_df["battery"] < battery_limit]
        elif "battery <=" in part:
            battery_limit = int(part.split("<=")[1].strip())
            result_df = result_df[result_df["battery"] <= battery_limit]
        elif "battery >" in part:
            battery_limit = int(part.split(">")[1].strip())
            result_df = result_df[result_df["battery"] > battery_limit]
        elif "battery >=" in part:
            battery_limit = int(part.split(">=")[1].strip())
            result_df = result_df[result_df["battery"] >= battery_limit]
        elif "battery DESC" in part:
            result_df = result_df.sort_values(by="battery", ascending=False)
        
        # Kamera filtreleri
        elif "camera <" in part:
            camera_limit = int(part.split("<")[1].strip())
            result_df = result_df[result_df["camera"] < camera_limit]
        elif "camera <=" in part:
            camera_limit = int(part.split("<=")[1].strip())
            result_df = result_df[result_df["camera"] <= camera_limit]
        elif "camera >" in part:
            camera_limit = int(part.split(">")[1].strip())
            result_df = result_df[result_df["camera"] > camera_limit]
        elif "camera >=" in part:
            camera_limit = int(part.split(">=")[1].strip())
            result_df = result_df[result_df["camera"] >= camera_limit]
        elif "camera DESC" in part:
            result_df = result_df.sort_values(by="camera", ascending=False)
    
    return result_df

def main():
    # CSV dosya yolunu tanımla (gerçek uygulamada bu bir argüman olabilir)
    csv_file = "phones.csv"
    
    # Kullanıcıdan prompt al
    user_prompt = input("Telefon kriterlerinizi doğal dille yazın (Örn: Fiyatı 10000 TL altında, RAM'i 6 GB üzeri): ")
    
    # Promptu filtre sorgusuna dönüştür
    filter_query = predict_filter_query(user_prompt)
    print(f"\nOluşturulan filtre sorgusu: {filter_query}")
    
    try:
        # CSV dosyasını filtrele
        filtered_phones = filter_phones_with_query(csv_file, filter_query)
        
        # Sonuçları göster
        print(f"\nBulundu: {len(filtered_phones)} telefon")
        if not filtered_phones.empty:
            print(filtered_phones.to_string(index=False))
        else:
            print("Kriterlere uygun telefon bulunamadı.")
    except FileNotFoundError:
        print(f"Hata: '{csv_file}' dosyası bulunamadı.")
        print("Lütfen 'phones.csv' dosyasını aynı dizinde olduğundan emin olun veya dosya yolunu düzenleyin.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()