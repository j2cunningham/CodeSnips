# app.py - a minimal flask api using flask_restful
import json
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    data = request.get_json()
    print(f'----------------{data}')
    return jsonify({'name': 'alice',
                    'email': 'alice@outlook.com'})

app.run()

if __name__ == '__main__':
    app.run(port=5001)

