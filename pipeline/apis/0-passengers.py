#!/usr/bin/env python3
"""Return the SWAPI starships able to carry a given number of passengers."""
import requests
import json


def availableShips(passengerCount):
    """Return the list of ships that can hold a given number of passengers."""

    # create page counter
    count = 1
    # API url without page count
    base = "https://swapi-api.hbtn.io/api/starships/"
    # API endpoint, remove trailing backslash
    api_url = base.rstrip('/') + f"/?page={count}"

    out = []
    while api_url:
        # fetch JSON
        r = requests.get(api_url)
        r.raise_for_status()
        response_json = r.text

        # convert JSON to Python object
        payload = json.loads(response_json)

        # extract data
        for ship in payload['results']:

            # skip n/a
            if ship['passengers'] in ('n/a', 'unknown'):
                continue

            # convert str to int
            string = ship['passengers'].replace(",", "")
            value = int(string)
            if value >= passengerCount:
                out.append(ship['name'])

        # JSON null is Python None
        if payload["next"] is None:
            return out
        else:
            # build the next API endpoint
            count += 1
            api_url = base.rstrip('/') + f'/?page={count}'

    return out
