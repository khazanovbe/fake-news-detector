import socket
import re
import json
import pickle
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

HOST = '127.0.0.1'
PORT = 8000

model = pickle.load(open('model.pkl', 'rb'))
vect = pickle.load(open('tfidfvect.pkl', 'rb'))

def detectNews(news):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', news)
    review = review.lower()
    review = review.split() 
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    valPkl = vect.transform([review]).toarray()
    prediction = model.predict(valPkl)

    return prediction[0]



with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f'Server listening on {HOST}:{PORT}...')

    while True:
        conn, addr = s.accept()
        with conn:
            print(f'Connected by {addr}')

            data = conn.recv(1024)
            request = data.decode('utf-8')
            method, path = re.findall(r'^([A-Z]+) (.*) HTTP/1.[01]', request)[0]
            if method == 'POST' and path == '/api/detect':
                
                request_body = request.split("\r\n\r\n")[1]

                data = json.loads(request_body)

                news = data['news']

                isTrue = detectNews(news)

                response_data = {'isTrue': bool(isTrue)}
                response = f'HTTP/1.1 200 OK\nContent-Type: application/json\n\n{json.dumps(response_data)}'
                conn.sendall(response.encode('utf-8'))
            else:
                conn.sendall(b'HTTP/1.1 404 Not Found\nContent-Type: text/html\n\nNot Found')