from numpy import loadtxt, arange, array, where, max, abs, mean
from pandas import DataFrame, ExcelWriter, read_excel
from subroutines.Spectra import FourierAmplitude
import os

exp_list = [
    "A",
    "OT",
    "RC",
    "SP",
    "SP-OT",
    "SP-RC",
    "RC-SP",
    "OT-SP",
]  # Source sağdaymış gibi
layouts = ["L1", "L2", "L3", "L4", "L5"]
frequencies = arange(10, 160, 10)
scaleFactors = {10: 1, 20: 1.2, 30: 1.5, 40: 1.75}
distances = {
    "L1": [1, 1.5, 2, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
    "L2": [1, 2, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
    "L3": [1, 2.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
    "L4": [1, 4.75, 8.75, 11, 13.5, 16, 18.5, 21, 25, 29],
    "L5": [1, 6, 8.5, 11, 13.5, 16, 18.5, 25, 29],
}
patterns = {
    "L1": [1, 1.5, 2, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
    "L2": [1, 2, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
    "L3": [1, 2.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
    "L4": [1, 4.75, 8.75, 11, 13.5, 16, 18.5, 21, 25, 29],
    "L5": [1, 6, 8.5, 11, 13.5, 16, 18.5, 25, 29],
}
D2S = {
    "L1": 2.25,
    "L2": 4.75,
    "L3": 7.25,
    "L4": 9.75,
    "L5": 12.25,
}

indexes = {"near": [4, 8], "far": [12, 16]}


def run(func):
    arg_list = [
        ("Milas", "Acc"),
        ("Milas", "Vel"),
        ("Mentese", "Vel"),
        ("Mentese", "Acc"),
        ("Senaryo1", "Vel"),
        ("Senaryo1", "Acc"),
        ("Senaryo2", "Vel"),
        ("Senaryo2", "Acc"),
        ("Senaryo3", "Vel"),
        ("Senaryo3", "Acc"),
    ]

    processDict = {}
    import multiprocessing

    if __name__ == "__main__":
        for i, arg in enumerate(arg_list):
            processDict[i] = multiprocessing.Process(target=func, args=arg)

        for p in processDict:
            processDict[p].start()

        for p in processDict:
            processDict[p].join()


def scale(exp, frequency, data):
    if exp in ["SP-OT", "OT-SP", "SP-RC", "RC-SP"]:
        scaleFac = 1 / 1.2
    elif exp == "A" or exp == "SP":
        scaleFac = 1
    else:
        scaleFac = 1 / scaleFactors.get(frequency, 2)

    data[3:] = data[3:] * scaleFac
    return data


def read_file(model_name, dataType):
    path = f"D:\\Projeler\\Tubitak\\Output\\2D\\{model_name}"
    datas = loadtxt(
        os.path.join(path, f"{model_name}_{dataType}.txt"), skiprows=1, unpack=True
    )[1:]
    return datas


def readField(dataType, city, exp, pattern, frequency):
    fileName = f"{exp}-{frequency}-{pattern}_{dataType[:3]}.txt"
    path = f"F:\\DataFolder\\Baseline Corrected\\{dataType}\\1 - Time History\\{dataType} Time History\\{city}"
    datas = loadtxt(os.path.join(path, fileName), unpack=True)
    return datas


def path_creator(path):
    path = os.path.join("Datas", "NewGraphs", "Filtered", path)
    if not os.path.exists(path):
        os.makedirs(path)


def fourier(datas):
    fa_list = []
    for data in datas:
        f, fa, _ = FourierAmplitude(data, 0.0005)
        fa_list.append(fa)
    return fa_list


def maxFunc(dataType, city, exp, pattern, frequency):
    model_name = f"{city}_{exp}_{pattern}_{frequency}Hz_2D"
    data = read_file(model_name, dataType)
    aranged = sorted(max(abs(data), axis=1), reverse=True)
    return scale(exp, frequency, array(aranged, dtype=float))


def DFFunc(dataType, city, exp, pattern):
    Max, NA, AR = {}, {}, {}
    for F in frequencies:
        barrierMax = maxFunc(dataType, city, exp, pattern, F)
        Max[F] = array(barrierMax)
        normalised = Max[F] / Max[F][0]
        NA[F] = normalised
        max_A = maxFunc(dataType, city, "A", pattern, F)
        NA_A = max_A / max_A[0]

        ar = normalised / NA_A
        AR[F] = ar

    DF_max = DataFrame.from_dict(
        Max, orient="index", columns=[f"C{i+1}" for i in range(len(Max[10]))]
    )
    DF_NA = DataFrame.from_dict(
        NA, orient="index", columns=[f"C{i+1}" for i in range(len(NA[10]))]
    )
    DF_AR = DataFrame.from_dict(
        AR, orient="index", columns=[f"C{i+1}" for i in range(len(AR[10]))]
    )
    return DF_max, DF_NA, DF_AR


def toExcel(city, dataType):
    for pattern in ["L1", "L2", "L3", "L4", "L5"]:
        Max_path = (
            f"D:\\Projeler\\Tubitak\\Output\\Excel\\{dataType}\\2 - Max Values\\{city}"
        )
        NA_path = f"D:\\Projeler\\Tubitak\\Output\\Excel\\{dataType}\\3 - Normalized Values\\{city}"
        AR_path = f"D:\\Projeler\\Tubitak\\Output\\Excel\\{dataType}\\4 - AR(Conventional Method)\\{city}"
        path_creator(Max_path)
        path_creator(NA_path)
        path_creator(AR_path)
        writer_max = ExcelWriter(
            os.path.join(Max_path, f"{pattern}.xlsx"), engine="xlsxwriter"
        )
        writer_NA = ExcelWriter(
            os.path.join(NA_path, f"{pattern}.xlsx"), engine="xlsxwriter"
        )
        writer_AR = ExcelWriter(
            os.path.join(AR_path, f"{pattern}.xlsx"), engine="xlsxwriter"
        )
        for exp in ["A", "OT", "RC", "SP", "SP-OT", "SP-RC", "RC-SP", "OT-SP"]:
            print(dataType, city, f"{exp}-{pattern}", dataType)
            Max, NA, AR = DFFunc(dataType, city, exp, pattern)
            Max.to_excel(writer_max, sheet_name=exp)
            NA.to_excel(writer_NA, sheet_name=exp)
            AR.to_excel(writer_AR, sheet_name=exp)

        writer_max.save()
        writer_NA.save()
        writer_AR.save()


def direct(secondSensor, dataType, location, pattern, exp, frequency):
    # NA_barrier = readField(dataType,location,pattern,exp,frequency)
    NA_barrier = maxFunc(dataType, location, exp, pattern, frequency)
    # NA_attenuation = readField(dataType,location,pattern,"A",frequency)
    NA_attenuation = maxFunc(dataType, location, "A", pattern, frequency)
    AR_list = []

    for firstSensor in [1, 2, 3]:
        AR = (NA_barrier[secondSensor - 1] * NA_attenuation[firstSensor - 1]) / (
            NA_barrier[firstSensor - 1] * NA_attenuation[secondSensor - 1]
        )
        AR_list.append(AR)
    AR_list.append(mean(AR_list))
    return AR_list


def DFFuncDirect(dataType, city, exp, pattern):
    AR = {}
    startFrom = 5 if pattern == "L5" else 4
    seconds = arange(startFrom, len(distances[pattern]) + 1)
    for f in frequencies:
        columns = []
        temp = []
        for s in seconds:
            columns.extend([f"S{s}(1)", f"S{s}(2)", f"S{s}(3)", f"S{s}(avg)"])
            temp.extend(direct(s, dataType, city, pattern, exp, f))
        AR[f] = temp
    DF_AR = DataFrame.from_dict(AR, orient="index", columns=columns)
    return DF_AR


def toExcelDirect(city, dataType):
    path = f"D:\Projeler\Tubitak\Output\Excel\{dataType}\\5 - AR(Direct Method)\\{city}"
    path_creator(path)
    for pattern in ["L1", "L2", "L3", "L4", "L5"]:
        fileName = f"{pattern}.xlsx"
        writer_AR = ExcelWriter(os.path.join(path, fileName), engine="xlsxwriter")
        for exp in ["OT", "RC", "SP", "SP-OT", "SP-RC", "OT-SP", "RC-SP"]:
            print(dataType, city, f"{exp}-{pattern}", dataType)
            AR = DFFuncDirect(dataType, city, exp, pattern)
            AR.to_excel(writer_AR, sheet_name=f"{exp}")

        writer_AR.save()


def readAR(dataType, location, pattern, exp, region):
    path = f"D:\\Projeler\\Tubitak\\Output\\Excel\\{dataType}\\5 - AR(Direct Method)\\{location}\\{pattern}.xlsx"
    data = []
    ind = indexes[region]
    for F in frequencies:
        excel = read_excel(path, sheet_name=f"{exp}")
        F_col = excel.iloc[:, 0]
        index = where(F_col == F)[0][0]
        data.append(mean(excel.iloc[index, ind].values))

    return data


def DFFuncAvg(city, dataType, exp, region):
    AR = {}
    for L in layouts:
        AR[L] = readAR(dataType, city, L, exp, region)

    DF = DataFrame.from_dict(AR, orient="index", columns=frequencies)
    return DF


def toExcelAvg(city, dataType):
    path = f"D:\\Projeler\\Tubitak\\Output\\Excel\\{dataType}\\6 - AR Direct (Avg)"
    print(city, dataType)
    path_creator(path)
    for region in ["far", "near"]:
        writer = ExcelWriter(
            os.path.join(path, f"{city}-{region}.xlsx"), engine="xlsxwriter"
        )
        for exp in exp_list[1:]:
            DF = DFFuncAvg(city, dataType, exp, region)
            DF.to_excel(writer, sheet_name=exp)
        writer.save()


# run(toExcelDirect)
run(toExcelAvg)
