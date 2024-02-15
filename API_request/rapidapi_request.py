# Imports
import pandas as pd
import numpy as np
import json

# pour requêtes api
import http.client
import requests


def get_api_data(ingredient):

    url = 'https://edamam-food-and-grocery-database.p.rapidapi.com/api/food-database/v2/parser'

    # Query
    querystring = {'ingr' : {ingredient}}

    # Credentials
    headers = {
        'X-RapidAPI-Key': '820c106fafmshd852fcb9c53095ap1d02cajsn7d59f3799ed5',
        'X-RapidAPI-Host': 'edamam-food-and-grocery-database.p.rapidapi.com'
    }

    # Réponse
    response = requests.get(url, headers = headers, params = querystring)
    result = response.json()
    df = pd.DataFrame(result['hints'])
    df = df['food'].apply(pd.Series)


    return df