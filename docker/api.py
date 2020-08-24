# app.py - a minimal flask api using flask_restful
import json
from flask import Flask, jsonify, json, Response, request
import pickle
import pandas as pd 

# class ScoringService(object):

#     model = None                # Where we keep the model when it's loaded

#     @classmethod
#     def get_model(cls):
#         if cls.model == None:
#             with open(os.path.join(model_path, 'random-forest-model.pkl'), 'rb') as inp:
#                 cls.model = pickle.load(inp)
#         return cls.model


#     @classmethod
#     def predict(cls, input):
#         clf = cls.get_model()
#         return clf.predict(input)


app = Flask(__name__)

# clf = pickle.load(open('model location', 'rb'))

@app.route('/', methods=['POST'])
def index():
    data = request.get_json()
    print(f'----------------{data}')

    # if flask.request.content_type.startswith('text/csv'):
    #     data = flask.request.data.decode('utf-8')
    #     s = io.StringIO(data)
    #     data = pd.read_csv(s, header=None)
    #     print('Invoked with {} batch records'.format(data.shape[0]))
    #     predictions = ScoringService.predict(data)
    #     out = io.StringIO()
    #     pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    #     result = out.getvalue()
    # elif flask.request.content_type.startswith('application/json'):
    #     data = flask.request.json
    #     data = pd.DataFrame.from_dict(data, orient="index")
    #     print('Invoked with {} api records'.format(data.T.shape[0]))
    #     predictions = ScoringService.predict(data.T)
    #     output = str(predictions[0])
    #     result = json.dumps({'result':output})
    # else:
    #     return flask.Response(response='This Predictor only supports CSV or json data', status=415, mimetype='text/plain')


    # return flask.Response(response=result, status=200, mimetype='text/csv')
    return data

@app.route('/health',methods=['GET'])
def health():
    js = json.dumps({'status':'healthy'})
    resp = Respone(js, status=200, mimetype='application/json')
    return resp 

if __name__ == '__main__':
    app.run(host='0.0.0.0')

