from pandas import read_excel, DataFrame, ExcelWriter
from numpy import array, arange, where, mean
import os

frequencies = arange(10, 160, 10)
mainFolder = "Filtered&Trimmed"


def path_creator(path):
    if not os.path.exists(path):
        os.makedirs(path)


def readField(dataType, city, pattern, exp, frequency, outputType):
    path = f"F:\\DataFolder\\{mainFolder}\\{dataType}\\2 - Max Values\\{city}\\Max {outputType}.xlsx"
    excel = read_excel(path, sheet_name=f"{exp}-{pattern}")
    F_col = excel.iloc[:, 0]
    index = where(F_col == frequency)[0][0]
    return excel.iloc[index, 1:-1].values


def direct(secondSensor, dataType, location, pattern, exp, frequency, outputType):
    NA_barrier = readField(dataType, location, pattern, exp, frequency, outputType)
    NA_attenuation = readField(dataType, location, pattern, "A", frequency, outputType)
    AR_list = []
    for firstSensor in [1, 2, 3]:
        AR = (NA_barrier[secondSensor - 1] * NA_attenuation[firstSensor - 1]) / (
            NA_barrier[firstSensor - 1] * NA_attenuation[secondSensor - 1]
        )
        AR_list.append(AR)
    AR_list.append(mean(AR_list))
    return AR_list


def DFFunc(dataType, city, exp, pattern, outputType):
    AR = {}
    for f in frequencies:
        columns = []
        temp = []
        for s in [4, 5, 6, 7, 8]:
            columns.extend([f"S{s}(1)", f"S{s}(2)", f"S{s}(3)", f"S{s}(avg)"])
            temp.extend(direct(s, dataType, city, pattern, exp, f, outputType))
        AR[f] = temp
    DF_AR = DataFrame.from_dict(AR, orient="index", columns=columns)
    return DF_AR


def toExcel(city, dataType, outputType):
    path = f"F:\\DataFolder\\{mainFolder}\\{dataType}\\5 - AR(Direct Method)\\{city}"
    path_creator(path)
    fileName = f"{outputType} (1, 2, 3 and Average).xlsx"
    writer_AR = ExcelWriter(os.path.join(path, fileName), engine="xlsxwriter")
    for exp in ["OT", "RC", "SP", "SP-OT", "SP-RC"]:
        for pattern in ["L25", "L50"]:
            print(dataType, city, f"{exp}-{pattern}", dataType)
            AR = DFFunc(dataType, city, exp, pattern, outputType)
            AR.to_excel(writer_AR, sheet_name=f"{exp}-{pattern}")

    writer_AR.save()


arg_list = [
    ("Milas", "Velocity", "Velocity"),
    ("Milas", "Velocity", "Fourier"),
    ("Milas", "Velocity", "Displacement"),
    ("Milas", "Acceleration", "Acceleration"),
    ("Milas", "Acceleration", "Fourier"),
    ("Milas", "Acceleration", "Displacement"),
    ("Menteşe", "Velocity", "Velocity"),
    ("Menteşe", "Velocity", "Fourier"),
    ("Menteşe", "Velocity", "Displacement"),
    ("Menteşe", "Acceleration", "Acceleration"),
    ("Menteşe", "Acceleration", "Fourier"),
    ("Menteşe", "Acceleration", "Displacement"),
]

func = toExcel
processDict = {}
import multiprocessing

if __name__ == "__main__":
    for i, arg in enumerate(arg_list):
        processDict[i] = multiprocessing.Process(target=func, args=arg)

    for p in processDict:
        processDict[p].start()

    for p in processDict:
        processDict[p].join()
