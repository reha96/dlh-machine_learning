#!/usr/bin/env python3
"""Return the SWAPI starships able to carry a given number of passengers."""
import requests


def availableShips(passengerCount):
    """Return the list of ships that can hold a given number of passengers."""
    api_url = "https://swapi-api.hbtn.io/api/starships/"

    out = []

    r = requests.get(api_url)
    r.raise_for_status()

    while api_url:
        r = requests.get(api_url)
        r.raise_for_status()
        response_json = r.json

        # build the next API endpoint
        count += 1
        api_url = base_url.rstrip('/') + f'/api/quotes?page={count}'

    return out