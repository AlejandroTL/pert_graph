"""
Load the config file and create any custom variables
that are available for ease of use purposes
"""

import yaml
import os
import sys


with open("/home/aletl/Documents/config/config.sample.yml", "r") as configFile:
    data = configFile.read()

data = yaml.load(data, Loader=yaml.FullLoader)

# ACCESS KEY
ACCESS_KEY = data["biogrid"]["access_key"]
BASE_URL = data["biogrid"]["base_url"]
