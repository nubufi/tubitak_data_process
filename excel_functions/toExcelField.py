from numpy import loadtxt, arange, array, zeros, max, abs
from numpy.lib.twodim_base import flipud
from pandas import DataFrame, ExcelWriter
from itertools import product
from subroutines.Spectra import FourierAmplitude
from shutil import move
import os

mainFolder = "Filtered&Trimmed"


def readField(dataType, city, exp, pattern, frequency, outputType):
    outputFolder = (
        "Fourier Spectrum" if outputType == "Fourier" else f"{outputType} Time History"
    )
    if (
        outputType == "Fourier"
        or outputType == "Displacement"
        or mainFolder == "Filtered&Trimmed"
    ):
        fileName = f"{frequency}Hz.txt"
    else:
        fileName = f"{exp}-{frequency}-{pattern}_{dataType[:3]}.txt"
    path = f"F:\\DataFolder\\{mainFolder}\\{dataType}\\1 - Time History\\{outputFolder}\\{city}\\{exp}-{pattern}"
    datas = loadtxt(os.path.join(path, fileName), unpack=True)
    return datas[1:] if outputType == "Fourier" else datas


def path_creator(path):
    if not os.path.exists(path):
        os.makedirs(path)


def fourier(datas):
    fa_list = []
    for data in datas:
        f, fa, _ = FourierAmplitude(data, 0.0005)
        fa_list.append(fa)
    return fa_list


def maxFunc(dataType, city, exp, pattern, frequency, outputType):
    data = readField(dataType, city, exp, pattern, frequency, outputType)
    return max(abs(data), axis=1)


def DFFunc(dataType, city, exp, pattern, outputType):
    Max, NA, AR = {}, {}, {}
    for F in frequencies:
        barrierMax = maxFunc(dataType, city, exp, pattern, F, outputType)
        Max[F] = array(barrierMax)

        normalised = Max[F] / Max[F][0]
        NA[F] = normalised

        max_A = maxFunc(dataType, city, "A", pattern, F, outputType)
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


def toExcel(city, dataType, outputType):
    Max_path = f"F:\\DataFolder\\{mainFolder}\\{dataType}\\2 - Max Values\\{city}"
    NA_path = f"F:\\DataFolder\\{mainFolder}\\{dataType}\\3 - Normalized Values\\{city}"
    AR_path = (
        f"F:\\DataFolder\\{mainFolder}\\{dataType}\\4 - AR(Conventional Method)\\{city}"
    )
    path_creator(Max_path)
    path_creator(NA_path)
    path_creator(AR_path)
    writer_max = ExcelWriter(
        os.path.join(Max_path, f"Max {outputType}.xlsx"), engine="xlsxwriter"
    )
    writer_NA = ExcelWriter(
        os.path.join(NA_path, f"Normalized {outputType} - S1.xlsx"), engine="xlsxwriter"
    )
    writer_AR = ExcelWriter(
        os.path.join(AR_path, f"{outputType} - S1.xlsx"), engine="xlsxwriter"
    )
    for pattern in ["L25", "L50"]:
        for exp in ["A", "OT", "RC", "SP", "SP-OT", "SP-RC"]:
            print(dataType, city, f"{exp}-{pattern}", outputType)
            Max, NA, AR = DFFunc(dataType, city, exp, pattern, outputType)
            Max.to_excel(writer_max, sheet_name=f"{exp}-{pattern}")
            NA.to_excel(writer_NA, sheet_name=f"{exp}-{pattern}")
            AR.to_excel(writer_AR, sheet_name=f"{exp}-{pattern}")

    writer_max.save()
    writer_NA.save()
    writer_AR.save()


def export2excel_records(exp, city, pattern, folder, mainfolder):
    writer_acc = ExcelWriter(
        "D:\\Projeler\\Tubitak\\Kodlar\\Datas\\Exceller\\Numeric\\{}\\{}\\Accelerations\\{}-{}.xlsx".format(
            mainfolder, city, exp, pattern
        ),
        engine="xlsxwriter",
    )
    writer_vel = ExcelWriter(
        "D:\\Projeler\\Tubitak\\Kodlar\\Datas\\Exceller\\Numeric\\{}\\{}\\Velocities\\{}-{}.xlsx".format(
            mainfolder, city, exp, pattern
        ),
        engine="xlsxwriter",
    )
    os.chdir(os.path.join(mainfolder, city, folder))
    for fr in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
        if exp == "DT":
            AccFileName = "{}-{}_Acc.txt".format(exp, str(fr))
            VelFileName = "{}-{}_Vel.txt".format(exp, str(fr))
        else:
            AccFileName = "{}-{}-{}_Acc.txt".format(exp, str(fr), pattern)
            VelFileName = "{}-{}-{}_Vel.txt".format(exp, str(fr), pattern)

        if AccFileName in os.listdir("Accelerations"):
            accelerations = read_file(
                os.path.join("Accelerations", AccFileName),
                [
                    [0, 0.5],
                    [0, 0.5],
                    [0, 0.5],
                    [0, 0.5],
                    [0, 0.5],
                    [0, 0.5],
                    [0.89, 2.5],
                    [0.89, 2.5],
                    [0.89, 2.5],
                ],
            )
            velocities = read_file(
                os.path.join("Velocities", VelFileName),
                [
                    [0, 3.47],
                    [0, 3.47],
                    [0, 3.47],
                    [0, 3.47],
                    [0, 3.47],
                    [0, 3.47],
                    [0, 3.47],
                    [0, 3.47],
                    [0, 3.47],
                ],
            )
            Sheet_acc = {}
            Sheet_vel = {}
            for i in range(len(accelerations)):
                Sheet_acc["Accelerometer {}(g)".format(i + 1)] = accelerations[i]
                # Sheet_vel["Accelerometer {}(cm/s)".format(i+1)] = acc2vel(accelerations[i])
            for i in range(len(velocities)):
                Sheet_acc["Geophone {}(g)".format(i + 1)] = vel2acc(velocities[i])
                # Sheet_vel["Geophone {}(cm/s)".format(i+1)] = velocities[i]

            DF_acc = DataFrame(Sheet_acc)
            DF_acc.to_excel(writer_acc, sheet_name=str(fr), index=False)

            # DF_vel = DataFrame(Sheet_vel)
            # DF_vel.to_excel(writer_vel,sheet_name=str(fr),index=False)

    writer_acc.save()
    writer_vel.save()
    os.chdir("..")
    os.chdir("..")
    os.chdir("..")


def export2excel_normalized(city):
    sourceFolder = "D:\\Projeler\\Tubitak\\Datas"
    outputFolder = "D:\\Projeler\\Tubitak\\Output"
    path_creator(
        os.path.join(sourceFolder, "Exceller", "Numeric", city, "Accelerations")
    )
    path_creator(os.path.join(sourceFolder, "Exceller", "Numeric", city, "Velocities"))
    path_creator(os.path.join(sourceFolder, "Exceller", "Numeric", city, "Fourier"))

    writer_acc_max = ExcelWriter(
        os.path.join(
            sourceFolder, "Exceller", "Numeric", city, "Accelerations", "MaxValues.xlsx"
        ),
        engine="xlsxwriter",
    )
    writer_acc_na = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Accelerations",
            "NormalizedValues.xlsx",
        ),
        engine="xlsxwriter",
    )
    writer_acc_ar = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Accelerations",
            "AmplitudeReduction.xlsx",
        ),
        engine="xlsxwriter",
    )

    writer_vel_max = ExcelWriter(
        os.path.join(
            sourceFolder, "Exceller", "Numeric", city, "Velocities", "MaxValues.xlsx"
        ),
        engine="xlsxwriter",
    )
    writer_vel_na = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Velocities",
            "NormalizedValues.xlsx",
        ),
        engine="xlsxwriter",
    )
    writer_vel_ar = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Velocities",
            "AmplitudeReduction.xlsx",
        ),
        engine="xlsxwriter",
    )

    writer_Accfourier_max = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Accelerations",
            "MaxValues_Fourier.xlsx",
        ),
        engine="xlsxwriter",
    )
    writer_Accfourier_na = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Accelerations",
            "NormalizedValues_Fourier.xlsx",
        ),
        engine="xlsxwriter",
    )
    writer_Accfourier_ar = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Accelerations",
            "AmplitudeReduction_Fourier.xlsx",
        ),
        engine="xlsxwriter",
    )

    writer_Velfourier_max = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Velocities",
            "MaxValues_Fourier.xlsx",
        ),
        engine="xlsxwriter",
    )
    writer_Velfourier_na = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Velocities",
            "NormalizedValues_Fourier.xlsx",
        ),
        engine="xlsxwriter",
    )
    writer_Velfourier_ar = ExcelWriter(
        os.path.join(
            sourceFolder,
            "Exceller",
            "Numeric",
            city,
            "Velocities",
            "AmplitudeReduction_Fourier.xlsx",
        ),
        engine="xlsxwriter",
    )

    sheet_acc_max = {}
    sheet_acc_na = {}
    sheet_acc_ar = {}

    sheet_accFourier_max = {}
    sheet_accFourier_na = {}
    sheet_accFourier_ar = {}

    sheet_vel_max = {}
    sheet_vel_na = {}
    sheet_vel_ar = {}

    sheet_velFourier_max = {}
    sheet_velFourier_na = {}
    sheet_velFourier_ar = {}

    exp_list = ["A", "OT", "RC", "SP", "SP-OT", "SP-RC", "RC-SP", "OT-SP"]
    layouts = ["L1", "L2", "L3", "L4", "L5"]
    patterns = {
        "L1": [1, 1.5, 2, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
        "L2": [1, 2, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
        "L3": [1, 2.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
        "L4": [1, 4.75, 8.75, 11, 13.5, 16, 18.5, 21, 25, 29],
        "L5": [1, 6, 8.5, 11, 13.5, 16, 18.5, 25, 29],
        "L6": [1, 2.5, 6, 8.5, 11, 13.5, 16, 18.5, 21],
    }
    frequencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    n = 1
    total = len(exp_list) * len(layouts) * len(frequencies)
    for exp in exp_list:
        for layout in layouts:
            pexp = "{}-{}".format(exp, layout)
            for fr in frequencies:
                folder_name = f"{city}_{exp}_{layout}_{fr}Hz_2D_Pressure"
                AccFileName = os.path.join(
                    outputFolder, folder_name, f"{folder_name}_Acc.txt"
                )
                VelFileName = os.path.join(
                    outputFolder, folder_name, f"{folder_name}_Vel.txt"
                )

                A_AccFileName = os.path.join(
                    outputFolder,
                    folder_name.replace(exp, "A"),
                    "{}_Acc.txt".format(folder_name.replace(exp, "A")),
                )
                A_VelFileName = os.path.join(
                    outputFolder,
                    folder_name.replace(exp, "A"),
                    "{}_Vel.txt".format(folder_name.replace(exp, "A")),
                )
                if os.path.exists(AccFileName):
                    accelerations = read_file(AccFileName)
                    velocities = read_file(VelFileName)
                    acc_fourier = fourier(accelerations)
                    vel_fourier = fourier(velocities)

                    A_accelerations = read_file(A_AccFileName)
                    A_velocities = read_file(A_VelFileName)
                    A_acc_fourier = fourier(A_accelerations)
                    A_vel_fourier = fourier(A_velocities)

                    A_max_acc = array([max(abs(i)) for i in A_accelerations])
                    A_NA_acc = A_max_acc / A_max_acc[0]

                    A_max_accFourier = array([max(abs(i)) for i in A_acc_fourier])
                    A_NA_accFourier = A_max_accFourier / A_max_accFourier[0]

                    A_max_velFourier = array([max(abs(i)) for i in A_vel_fourier])
                    A_NA_velFourier = A_max_velFourier / A_max_velFourier[0]

                    A_max_vel = array([max(abs(i)) for i in A_velocities])
                    A_NA_vel = A_max_vel / A_max_vel[0]

                    max_acc = array([max(abs(i)) for i in accelerations])
                    NA_acc = max_acc / max_acc[0]
                    AR_acc = NA_acc / A_NA_acc

                    max_accFourier = array([max(abs(i)) for i in acc_fourier])
                    NA_accFourier = max_accFourier / max_accFourier[0]
                    AR_accFourier = NA_accFourier / A_NA_accFourier

                    max_vel = array([max(abs(i)) for i in velocities])
                    NA_vel = max_vel / max_vel[0]
                    AR_vel = NA_vel / A_NA_vel

                    max_velFourier = array([max(abs(i)) for i in vel_fourier])
                    NA_velFourier = max_velFourier / max_velFourier[0]
                    AR_velFourier = NA_velFourier / A_NA_velFourier

                    sheet_acc_max["{}".format(fr)] = max_acc
                    sheet_acc_na["{}".format(fr)] = NA_acc
                    sheet_acc_ar["{}".format(fr)] = AR_acc

                    sheet_accFourier_max["{}".format(fr)] = max_accFourier
                    sheet_accFourier_na["{}".format(fr)] = NA_accFourier
                    sheet_accFourier_ar["{}".format(fr)] = AR_accFourier

                    sheet_vel_max["{}".format(fr)] = max_vel
                    sheet_vel_na["{}".format(fr)] = NA_vel
                    sheet_vel_ar["{}".format(fr)] = AR_vel

                    sheet_velFourier_max["{}".format(fr)] = max_velFourier
                    sheet_velFourier_na["{}".format(fr)] = NA_velFourier
                    sheet_velFourier_ar["{}".format(fr)] = AR_velFourier
                    # graph_path = os.path.join(sourceFolder,"NewGraphs","Filtered",city,pexp,str(t),str(fr))

                    # AR_graph(p, AR_acc, graph_path, exp, t, "Acceleration")
                    # AR_graph(p, AR_vel, graph_path, exp, t, "Velocity")
                    # AR_graph(p, AR_accFourier, graph_path, exp, t, "Fourier(Acc)")
                    # AR_graph(p, AR_velFourier, graph_path, exp, t, "Fourier(Vel)")
                else:
                    sheet_acc_max["{}".format(fr)] = zeros(len(patterns[layout]))
                    sheet_acc_na["{}".format(fr)] = zeros(len(patterns[layout]))
                    sheet_acc_ar["{}".format(fr)] = zeros(len(patterns[layout]))

                    sheet_accFourier_max["{}".format(fr)] = zeros(len(patterns[layout]))
                    sheet_accFourier_na["{}".format(fr)] = zeros(len(patterns[layout]))
                    sheet_accFourier_ar["{}".format(fr)] = zeros(len(patterns[layout]))

                    sheet_vel_max["{}".format(fr)] = zeros(len(patterns[layout]))
                    sheet_vel_na["{}".format(fr)] = zeros(len(patterns[layout]))
                    sheet_vel_ar["{}".format(fr)] = zeros(len(patterns[layout]))

                    sheet_velFourier_max["{}".format(fr)] = zeros(len(patterns[layout]))
                    sheet_velFourier_na["{}".format(fr)] = zeros(len(patterns[layout]))
                    sheet_velFourier_ar["{}".format(fr)] = zeros(len(patterns[layout]))
                print("{}/{}".format(n, total))
                n += 1
            if exp != "A":
                DF_acc_ar = DataFrame.from_dict(
                    sheet_acc_ar,
                    orient="index",
                    columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
                )
                DF_acc_ar.to_excel(writer_acc_ar, sheet_name=pexp)
                DF_vel_ar = DataFrame.from_dict(
                    sheet_vel_ar,
                    orient="index",
                    columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
                )
                DF_vel_ar.to_excel(writer_vel_ar, sheet_name=pexp)
                DF_velFourier_ar = DataFrame.from_dict(
                    sheet_velFourier_ar,
                    orient="index",
                    columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
                )
                DF_velFourier_ar.to_excel(writer_Velfourier_ar, sheet_name=pexp)
                DF_accFourier_ar = DataFrame.from_dict(
                    sheet_accFourier_ar,
                    orient="index",
                    columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
                )
                DF_accFourier_ar.to_excel(writer_Accfourier_ar, sheet_name=pexp)

            # print(folder_name)
            # print(list(map(lambda x:len(x),sheet_acc_max.values())))
            DF_acc_max = DataFrame.from_dict(
                sheet_acc_max,
                orient="index",
                columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
            )
            DF_acc_max.to_excel(writer_acc_max, sheet_name=pexp)

            DF_acc_na = DataFrame.from_dict(
                sheet_acc_na,
                orient="index",
                columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
            )
            DF_acc_na.to_excel(writer_acc_na, sheet_name=pexp)

            DF_vel_max = DataFrame.from_dict(
                sheet_vel_max,
                orient="index",
                columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
            )
            DF_vel_max.to_excel(writer_vel_max, sheet_name=pexp)

            DF_vel_na = DataFrame.from_dict(
                sheet_vel_na,
                orient="index",
                columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
            )
            DF_vel_na.to_excel(writer_vel_na, sheet_name=pexp)

            DF_velFourier_max = DataFrame.from_dict(
                sheet_velFourier_max,
                orient="index",
                columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
            )
            DF_velFourier_max.to_excel(writer_Velfourier_max, sheet_name=pexp)

            DF_velFourier_na = DataFrame.from_dict(
                sheet_velFourier_na,
                orient="index",
                columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
            )
            DF_velFourier_na.to_excel(writer_Velfourier_na, sheet_name=pexp)

            DF_accFourier_max = DataFrame.from_dict(
                sheet_accFourier_max,
                orient="index",
                columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
            )
            DF_accFourier_max.to_excel(writer_Accfourier_max, sheet_name=pexp)

            DF_accFourier_na = DataFrame.from_dict(
                sheet_accFourier_na,
                orient="index",
                columns=[f"C{i+1}" for i in range(len(patterns[layout]))],
            )
            DF_accFourier_na.to_excel(writer_Accfourier_na, sheet_name=pexp)

    writer_acc_max.save()
    writer_acc_na.save()
    writer_acc_ar.save()
    writer_vel_max.save()
    writer_vel_na.save()
    writer_vel_ar.save()
    writer_Accfourier_max.save()
    writer_Accfourier_na.save()
    writer_Accfourier_ar.save()
    writer_Velfourier_max.save()
    writer_Velfourier_na.save()
    writer_Velfourier_ar.save()


exp_list = ["A", "OT", "RC", "SP", "SP-OT", "SP-RC"]  # Source sağdaymış gibi
layouts = ["L25", "L50"]
frequencies = arange(10, 160, 10)
locations = ["Menteşe", "Milas"]

arg_list = [
    ("Milas", "Acceleration", "Velocity"),
    ("Milas", "Velocity", "Acceleration"),
    ("Menteşe", "Acceleration", "Velocity"),
    ("Menteşe", "Velocity", "Acceleration"),
    # ("Milas", "Acceleration", "Acceleration"),
    # ("Milas", "Velocity", "Velocity"),
    # ("Menteşe", "Acceleration", "Acceleration"),
    # ("Menteşe", "Velocity", "Velocity"),
    # ("Milas", "Acceleration", "Displacement"),
    # ("Milas", "Velocity", "Displacement"),
    # ("Menteşe", "Acceleration", "Displacement"),
    # ("Menteşe", "Velocity", "Displacement"),
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
