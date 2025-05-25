def load_data(data_file):   
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    total = 0
    for line in lines:
        line = line.strip()
        if not line:  # boş satır kontrolü
            continue
        if line.startswith("//") or " -> " not in line:
            continue  # yorum satırı veya geçersiz satır atla
        input_text, output_text = line.split(" -> ", 1)
        total += 1
        print(input_text, output_text)
    print("total: ",total)
# Fonksiyon çağrısı
load_data("training_data.txt")
