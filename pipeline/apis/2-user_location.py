#!/usr/bin/env python3
"""Print the location of a GitHub user from the GitHub API."""
import sys
import json
import time

import requests


if __name__ == '__main__':
    # exit if no URL given
    if len(sys.argv) != 2:
        exit()
    # store URL as second arg of input
    url = sys.argv[1]

    # fetch user data
    r = requests.get(url, timeout=5)

    # handle missing user
    if r.status_code == 404:
        print("Not found")

    # handle rate limit
    elif r.status_code == 403:

        # read reset time from headers
        reset = int(r.headers["X-Ratelimit-Reset"])

        # get current time
        now = int(time.time())

        # convert seconds to minutes
        minutes = int((reset - now) / 60)
        print("Reset in {} min".format(minutes))

    # show location
    elif r.status_code == 200:

        # convert JSON to Python object
        response_json = r.text
        payload = json.loads(response_json)

        # print user location
        print(payload.get("location"))
