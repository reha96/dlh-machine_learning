#!/usr/bin/env python3
"""Return the home planets of all sentient species from the SWAPI API."""
import requests


def sentientPlanets():
    """Return the list of names of the home planets of all sentient species."""
    # start from the first species page
    api_url = "https://swapi-api.hbtn.io/api/species/"

    out = []
    while api_url:
        # fetch page
        r = requests.get(api_url)
        r.raise_for_status()
        payload = r.json()

        # extract data
        for species in payload['results']:

            # keep species sentient by classification or designation
            if species['classification'] == "sentient" or \
                    species['designation'] == "sentient":

                # skip species without a homeworld link
                if species['homeworld'] is None:
                    continue

                # follow homeworld link to planet name
                planet = requests.get(species['homeworld']).json()
                out.append(planet.get("name", "unknown"))

        # follow next page, None ends the loop
        api_url = payload["next"]

    return out
