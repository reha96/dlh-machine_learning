#!/usr/bin/env python3
"""Return the home planets of all sentient species from the SWAPI API."""
import requests
import json


def sentientPlanets():
    """Return the list of names of the home planets of all sentient species."""
    # create page counter
    count = 1
    # API url without page count
    base = "https://swapi-api.hbtn.io/api/species/"
    # API endpoint, remove trailing slash
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
        for species in payload['results']:
            # keep species sentient by classification or designation
            if species['classification'] == "sentient" or \
                    species['designation'] == "sentient":
                # skip species without a homeworld link
                if species['homeworld'] is None:
                    continue
                # fetch homeworld page
                home = requests.get(species['homeworld'])
                home.raise_for_status()
                planet = json.loads(home.text)
                # add planet name, unknown if missing
                out.append(planet.get("name", "unknown"))

        # JSON null is Python None
        if payload["next"] is None:
            return out
        else:
            # build the next API endpoint
            count += 1
            api_url = base.rstrip('/') + f'/?page={count}'

    return out
