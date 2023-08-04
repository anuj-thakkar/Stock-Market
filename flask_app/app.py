# create Flask app to predict Close price of stock

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
import datetime
import time
import requests

# import custom modules
from LSTM import feature_engineering, model_selection
