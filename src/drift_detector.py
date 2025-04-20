# drift_detector.py
# ADWIN / River module for drift detection

from river import drift

class DriftDetector:
    def __init__(self):
        self.adwin = drift.ADWIN()

    def update(self, value):
        self.adwin.update(value)
        return self.adwin.change_detected
