import numpy as np
from matplotlib import pyplot as plt
import requests, json

USER = "2d72ec3382"
API_KEY = "ff4bfb0c2d2fe8d64d0f523550567237617689c2cea999568bcf9f2f28d0c70e"

def get_news(user, api_key, lang):
    url = "https://algogene.com/rest/v1/realtime_news"
    headers = {'Content-Type':'application/json'}
    params = {'user':user, 'api_key':api_key, 'lang':lang}
    res = requests.get(url, params=params, headers=headers)
    return res.json()

data = get_news(USER, API_KEY, 'en')
print(data)