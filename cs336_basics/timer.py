from datetime import datetime, timedelta
import json

TIMEDELTA_0 = timedelta()

START_TIMES = {}

TIMES = {}

DEBUG = True

def start(name):
    if DEBUG:
        START_TIMES[name] = datetime.now()

def update(name):
    if DEBUG:
        TIMES[name] = TIMES.get(name, TIMEDELTA_0) + datetime.now() - START_TIMES[name]

def measure(name, fn):
    start(name)
    res = fn()
    update(name)
    return res

def get_times_str():
    return json.dumps({k:str(v) for k,v in TIMES.items()}, indent=4)