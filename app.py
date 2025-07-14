"""
App Simple Chat Bot
""" 
from flask import Flask, render_template, request
import json
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot',methods=["POST"])
def chatbot():
    data  = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']
    
    print(data)
    #create conversation histo
    history = "\n".join(conversation_history)

    #Tokenize input + histo
    input = tokenizer.encode_plus(history,input_text, return_tensors = 'pt')

    # response from model
    outputs = model.generate(**input, max_length=60)

    #decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    #add new conversation to history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run(debug = True)