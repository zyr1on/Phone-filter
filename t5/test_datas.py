 def load_data(self, data_file):   
	with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:  # boş satır kontrolü
            continue
        if line.startswith("//") or " -> " not in line:
            continue  # yorum satırı veya geçersiz satır atla
        input_text, output_text = line.split(" -> ", 1)
		print(input_text,output_text); 

load_data("training_data.txt")
