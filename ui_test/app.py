from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def parse_features(input_str):
    results = []
    # input'u ayır
    pairs = input_str.strip().split(';')
    # her key-value çiftini işle
    for pair in pairs:
        if ':' in pair:
            key, value = pair.split(':', 1)
            if not value.strip().lower() == "none":
                results.append(f"{key.strip()}:{value.strip()}")
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_input():
    user_input = request.json.get('input', '')
    
    if not user_input.strip():
        return jsonify({'response': 'Lütfen bir şey yazınız.'})
    
    parsed_results = parse_features(user_input)
    
    if not parsed_results:
        return jsonify({'response': 'Tekrar deneyiniz.'})
    
    response = '\n'.join(parsed_results)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
