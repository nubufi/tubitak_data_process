from subroutines.Spectra import FourierAmplitude
from subroutines.Time_Series import *
import os
import numpy as np

inputPath = "F:\\DataFolder\\Filtered&Trimmed"


def readFile(fileName, path):
    path = os.path.join(path, fileName)
    data = np.loadtxt(path, unpack=True)

    return data


def convert(data, inputType):
    if inputType == "Velocity":
        target, disp = fromVel(data)
    else:
        target, disp = fromAcc(data)

    return target, disp


def pathCreator(path):
    if not os.path.exists(path):
        os.makedirs(path)


def toTxt(data, fileName, outputPath):
    rowNo = len(data[0])
    path = os.path.join(outputPath, fileName)
    f = open(path, "w")
    for i in range(rowNo):
        delimiter = "\t" * 3
        line = delimiter.join([str(d[i]) for d in data]) + "\n"
        f.write(line)
    f.close()
