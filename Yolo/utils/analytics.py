import os
import json

class Analytics(object):

    # Singleton
    __instance = None
    def __new__(cls):
        if Analytics.__instance is None:
            Analytics.__instance = object.__new__(cls)

        return Analytics.__instance

    # If object is destroyed close files
    def __del__(self):
        try:
            self.epochLossFile.close()
        except Exception:
            pass

        try:
            self.validLossFile.close()
        except Exception:
            pass

    def __init__(self):
        os.makedirs("analytics", exist_ok=True)
        self.epochLossFile = open("analytics/epochLoss.txt", "w")
        self.validLossFile = open("analytics/validLoss.txt", "w")

    def LogEpochLoss(self, epoch, loss, numImgs):
        line = '%d %f %d\n' % (epoch, loss, numImgs)
        self.epochLossFile.write(line)

    def LogValidLoss(self, epoch, loss, numImgs):
        line = '%d %f %d\n' % (epoch, loss, numImgs)
        self.validLossFile.write(line)

    def logTestResults(self, epoch, resultDict):
        filename = "analytics/test_epoch_%d.json" % (epoch)
        f = open(filename, "w")
        json.dump(resultDict, f)
        f.close()
