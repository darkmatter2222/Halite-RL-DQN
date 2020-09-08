import time
import math
class stopwatch:
    def __init__(self):
        self.start()
    def start(self):
        self._startTime = time.time()
    def getStartTime(self):
        return self._startTime
    def elapsed(self, prec=3):
        prec = 3 if prec is None or not isinstance(prec, (int, int)) else prec
        diff= time.time() - self._startTime
        return round(diff, prec)

def round(n, p=0):
    m = 100 ** p
    return math.floor(n * m + 0.5) / m