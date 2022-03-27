from operator import le
from subroutines.PreProcess import Filtering
from subroutines.Spectra import FourierAmplitude
import os
from numpy import loadtxt

main_source_folder = os.path.join("Datas", "Texts", "BaselineCorrected")
frequencies = [10, 20, 30, 40, 50, 60, 70, 80, 90]

F2F = {
    10: [10],
    20: [20, 80, 70],
    30: [30, 60, 70, 80, 90, 120, 150],
    40: [40, 60, 80, 120],
    50: [50, 75, 100, 110, 120, 150],
    60: [60, 80, 120],
    70: [60, 70, 120, 110, 80],
    80: [80, 120],
    90: [80, 90],
}

dataTypes = ["Accelerations", "Velocities"]
patterns = ["L25", "L50"]
cities = ["Menteşe", "Milas"]
experiments = ["A", "OT", "RC", "SP", "SP-OT", "SP-RC", "DT"]

file = lambda c, e, d, f, p: os.path.join(
    main_source_folder,
    c,
    "{}-{}".format(e, p),
    d,
    "{}-{}-{}_{}.txt".format(e, f, p, d[:3]),
)


def path_creator(path):
    path = os.path.join("Datas", "Texts", "NewFiltered", path)
    if not os.path.exists(path):
        os.makedirs(path)


def read_file(file_name):
    data = loadtxt(file_name, skiprows=0, unpack=True)
    return data


def write2file(file_name, F1, F2):
    datas = read_file(file_name)
    filteredDatas = []
    for d in datas:
        f, fa, FP = FourierAmplitude(d, F2)
        filteredDatas.append(Filtering(d, 0.0005, FP - 2, FP + 2))

    folder = os.path.dirname(file_name).replace("BaselineCorrected", "NewFiltered")
    f = open(os.path.join(folder, "{}-{}.txt".format(F2, F1)), "w")
    for row in range(len(filteredDatas[0])):
        line = []
        for col in filteredDatas:
            line.append(str(col[row]))
        f.write("\t\t\t".join(line) + "\n")
    f.close()


total = len(cities) * len(experiments) * len(patterns) * len(dataTypes) * 33
i = 0
for c in cities:
    path_creator(c)
    for e in experiments:
        for p in patterns:
            path_creator(os.path.join(c, "{}-{}".format(e, p)))
            for d in dataTypes:
                path_creator(os.path.join(c, "{}-{}".format(e, p), d))
                for f in frequencies:
                    for t in F2F[f]:
                        try:
                            write2file(file(c, e, d, f, p), f, t)
                        except:
                            pass
                        i += 1
                        print("{}/{}".format(i, total))
