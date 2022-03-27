from pandas import read_excel, DataFrame, ExcelWriter
import numpy as np
import os

exp_list = ["A", "OT", "RC", "SP", "SP-OT", "SP-RC"]  # Source sağdaymış gibi
layouts = ["L2", "L3"]
frequencies = np.arange(10, 160, 10)
locations = ["Menteşe", "Milas"]


def path_creator(path):
    if not os.path.exists(path):
        os.makedirs(path)


patterns = {
    "L2": [1, 2, 3.5, 6, 8.5, 11, 13.5, 16],
    "L3": [1, 2.5, 6, 8.5, 11, 13.5, 16, 18.5],
}
D2S = {
    "L2": 4.75,
    "L3": 7.25,
}

indexes = {"near": [4, 8], "far": [12, 16]}


def readAR(dataType, location, pattern, exp, region):
    path = f"F:\\DataFolder\\Filtered&Trimmed\\{dataType}\\5 - AR(Direct Method)\\{location}\\{dataType} (1, 2, 3 and Average).xlsx"
    data = []
    ind = indexes[region]
    pattern = "L25" if pattern == "L2" else "L50"
    for F in frequencies:
        excel = read_excel(path, sheet_name=f"{exp}-{pattern}")
        F_col = excel.iloc[:, 0]
        index = np.where(F_col == F)[0][0]
        data.append(np.mean(excel.iloc[index, ind].values))

    return data


def DFFunc(city, dataType, exp, region):
    AR = {}
    for L in layouts:
        AR[L] = readAR(dataType, city, L, exp, region)

    DF = DataFrame.from_dict(AR, orient="index", columns=frequencies)
    return DF


def toExcel(city, dataType):
    path = f"F:\\DataFolder\\Filtered&Trimmed\\{dataType}\\6 - AR(Avg)"
    print(city, dataType)
    path_creator(path)
    for region in ["far", "near"]:
        writer = ExcelWriter(
            os.path.join(path, f"{city}-{region}.xlsx"), engine="xlsxwriter"
        )
        for exp in exp_list[1:]:
            DF = DFFunc(city, dataType, exp, region)
            DF.to_excel(writer, sheet_name=exp)
        writer.save()


def run():
    arg_list = [
        ("Milas", "Acceleration"),
        ("Milas", "Velocity"),
        ("Menteşe", "Velocity"),
        ("Menteşe", "Acceleration"),
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


run()
