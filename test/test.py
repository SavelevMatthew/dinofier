import requests

resp = requests.post('http://localhost:5000/names?amount=6&startsWith=abc')

print(resp.json())