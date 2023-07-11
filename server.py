from flask import Flask, request, jsonify

from utils import query_data, embed_OPENAI

app = Flask(__name__)

@app.route('/he')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

@app.post('/query')
def query_model():
    query = request.json['query']
    response = query_data(query=query)

    response_data = {
        "query" : query,
        "response" : response
    }

    return jsonify(response_data)



if __name__ == '__main__':
    
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)