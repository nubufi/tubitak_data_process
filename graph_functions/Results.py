from numpy import loadtxt, array, where, mean, arange, append, flipud
import matplotlib.pyplot as plt
import os
from pandas import read_excel
from subroutines.Spectra import FourierAmplitude


def read_file(file_name):
    datas = [loadtxt("{}".format(file_name), skiprows=1, unpack=True)][0][1:]
    return datas if max(datas[0]) > max(datas[-1]) else flipud(datas)


def path_creator(city, layout, exp, dataType, outputType, file_name):
    path = f"D:\\Projeler\\Tubitak\\Graphs\\Numeric\\{city}\\{exp}\\{layout}\\{dataType}\\{outputType}"
    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, f"{file_name}.png")


def fourier(datas):
    fa_list = []
    for data in datas:
        f, fa, _ = FourierAmplitude(data, 0.0005)
        fa_list.append(fa)

    max_fa = array([max(abs(i)) for i in fa_list])
    return max_fa / max_fa[0]


def NA_graph(output, ylabel, x, y, vline=False):
    plt.plot(x, y, marker="o")
    plt.semilogy()
    plt.ylabel(ylabel)
    plt.xlabel("Mesafe (m)")
    plt.xticks(arange(2, max(x) + 2, 2))
    plt.ylim([0.00001, 10])
    plt.grid(True)
    if vline:
        plt.vlines(vline, 0, 3)
        plt.text(
            vline, 3.1, "Dalga\nBariyeri", fontsize=9, horizontalalignment="center"
        )
    plt.savefig(output)
    # plt.show()
    plt.close()


def AR_graph(output, x, y, vline=False):
    plt.plot(x, y, marker="o")
    plt.ylabel("Titreşim Azalım Oranı")
    plt.xlabel("Mesafe (m)")
    plt.xticks(arange(2, max(x) + 2, 2))
    y_max = max(3, max(y))
    plt.ylim([0, y_max])
    plt.grid(True)
    if vline:
        plt.vlines(vline, 0, y_max - 0.5)
        plt.text(
            vline,
            y_max - 0.6,
            "Dalga\nBariyeri",
            fontsize=9,
            horizontalalignment="center",
        )
    plt.savefig(output)
    # plt.show()
    plt.close()


def trench_depth_NA(frequency_list, depth_list):
    data = {}
    for f in frequency_list:
        for d in depth_list:
            f_name = "O2_Hz{}_D{}".format(f, int(d * 10))
            data[f_name] = read_file(f_name)

    return data


def trench_depth_AR(frequency_list, depth_list, double_trench, free_field):
    data = {}
    for f in frequency_list:
        F_NA = free_field[f]
        for d in depth_list:
            f_name = "O2_Hz{}_D{}".format(f, int(d * 10))
            O2_NA = double_trench[f_name]

            data[f_name] = O2_NA / F_NA

    return data


def Ar_generator(depth, frequency_list, free_field, open_trench):
    data = {}
    for f in frequency_list:
        f_NA = free_field[f]
        O_NA = open_trench[f]
        D = depth * f / 212.85
        data[round(D, 2)] = O_NA / f_NA

    return data


def generate_graph(title, vertical_line, ylabel, xlabel, x, *accelerations):
    for i in accelerations:
        y = i["Acc"]
        x = i["X"]
        if i["type"] == "line":
            plt.plot(x, y, label=i["label"])
        else:
            plt.scatter(x, y, label=i["label"])
        plt.semilogy()
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(arange(2, 24, 2))
        plt.ylim([0.0001, 3])
        if vertical_line:
            for v in vertical_line:
                plt.text(
                    v, 1.1, "Dalga\nBariyeri", fontsize=9, horizontalalignment="center"
                )
                plt.vlines(v, 0, 1)
        # plt.title(title)
    plt.grid()
    plt.savefig(os.path.join("Grafikler", title))
    # plt.show()
    plt.close()


def channel_graph(title, ylabel, xlabel, accelerations, x):
    for i in range(len(accelerations[x[0]])):
        acc_list = []
        for n in x:
            acc_list.append(accelerations[n][i])
        plt.plot(x, array(acc_list) / max(acc_list), label="Channel {}".format(i + 1))
        plt.semilogy()
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid()
    plt.savefig(title)
    plt.show()
    plt.close()


def Ar_NormalizedDepth(title, field, abaqus, acc_num, vline=False):
    abaqus_x = list(abaqus.keys())
    abaqus_y = []
    for k in abaqus_x:
        acc = abaqus[k][acc_num]
        abaqus_y.append(acc)

    field_x = list(field.keys())
    field_y = []
    for k in field_x:
        acc = field[k][acc_num]
        field_y.append(acc)

    plt.plot(abaqus_x, abaqus_y, label="Abaqus")
    plt.scatter(field_x, field_y, label="Field")
    plt.semilogy()
    plt.grid()
    plt.legend()
    plt.title(title)
    if vline:
        for i in vline:
            plt.vlines(i, 0, max(list(abaqus_y) + list(field_y)))
    plt.ylabel("Amplitude Reduction Ratio")
    plt.xlabel("Normalized Depth")
    plt.savefig(title)
    plt.close()


def AR_Depth(title, abaqus, acc_num):
    abaqus_x = arange(2, 6.5, 0.5)
    for f in [50, 75, 100]:
        abaqus_y = []
        for d in abaqus_x:
            key = "O2_Hz{}_D{}".format(f, int(d * 10))
            acc = abaqus[key][acc_num]
            abaqus_y.append(acc)

        plt.plot(abaqus_x, abaqus_y, label="{} Hz".format(f))
    plt.semilogy()
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.ylabel("Amplitude Reduction Ratio")
    plt.xlabel("Trench Depth")
    plt.savefig(title)
    plt.close()


def fieldResults(page, row_label):
    excel = read_excel(
        "D:\\Projeler\\Tubitak\\Datas\\NewExceller\\Filtered\\Milas\\Accelerations\\NormalizedValues.xlsx",
        sheet_name=page,
    )
    labels = excel.iloc[:, 0]
    filter_ = labels == row_label
    return excel[filter_].iloc[0, 1:].values


n = 0


def drawGraph(exp, layout, frequency, location):
    global n
    outputFolder = "D:\Projeler\Tubitak\Output"
    folder_name = f"{location}_{exp}_{layout}_{frequency}Hz_2D_Pressure"
    AccFileName = os.path.join(outputFolder, folder_name, f"{folder_name}_Acc.txt")
    VelFileName = os.path.join(outputFolder, folder_name, f"{folder_name}_Vel.txt")
    acc = read_file(AccFileName)
    vel = read_file(VelFileName)
    NA_Func = lambda data: data / data[0]

    D2S = {
        "L1": 2.25 + 0.75 / 2,
        "L2": 4.75 + 0.75 / 2,
        "L3": 7.25 + 0.75 / 2,
        "L4": 9.75 + 0.75 / 2,
        "L5": 12.25 + 0.75 / 2,
    }

    patterns = {
        "L1": [1, 1.5, 2, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
        "L2": [1, 2, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
        "L3": [1, 2.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 25, 29],
        "L4": [1, 4.75, 8.75, 11, 13.5, 16, 18.5, 21, 25, 29],
        "L5": [1, 6, 8.5, 11, 13.5, 16, 18.5, 25, 29],
        "L6": [1, 2.5, 6, 8.5, 11, 13.5, 16, 18.5, 21],
    }

    vline = False if exp == "A" else D2S[layout]
    file_name = f"{exp}_{layout}_{frequency}Hz"

    # Normalized Acceleration
    max_acc = array([max(abs(i)) for i in acc])
    NA_acc = NA_Func(max_acc)
    outputPath = path_creator(location, layout, exp, "Acc", "Normalized", file_name)
    NA_graph(outputPath, "Normalize İvme", patterns[layout], NA_acc, vline)

    # Normalized Velocity
    max_vel = array([max(abs(i)) for i in vel])
    NA_vel = NA_Func(max_vel)
    outputPath = path_creator(location, layout, exp, "Vel", "Normalized", file_name)
    NA_graph(outputPath, "Normalize Hız", patterns[layout], NA_vel, vline)

    # Normalized Acceleration Fourier Amplitude
    NA_F_acc = fourier(acc)
    outputPath = path_creator(
        location, layout, exp, "Acc-Fourier", "Normalized", file_name
    )
    NA_graph(outputPath, "Normalize Fourier Genliği", patterns[layout], NA_F_acc, vline)

    # Normalized Velocity Fourier Amplitude
    NA_F_vel = fourier(vel)
    outputPath = path_creator(
        location, layout, exp, "Vel-Fourier", "Normalized", file_name
    )
    NA_graph(outputPath, "Normalize Fourier Genliği", patterns[layout], NA_F_vel, vline)

    n += 4

    # Amlitude Reduction Ratio
    if exp != "A":
        # Acceleration
        A_Acc = read_file(AccFileName.replace(exp, "A"))
        max_A_Acc = array([max(abs(i)) for i in A_Acc])
        NA_A_Acc = NA_Func(max_A_Acc)
        AR_acc = NA_acc / NA_A_Acc
        outputPath = path_creator(location, layout, exp, "Acc", "Ar", file_name)
        AR_graph(outputPath, patterns[layout], AR_acc, vline)

        # Velocity
        A_Vel = read_file(VelFileName.replace(exp, "A"))
        max_A_Vel = array([max(abs(i)) for i in A_Vel])
        NA_A_Vel = NA_Func(max_A_Vel)
        AR_Vel = NA_vel / NA_A_Vel
        outputPath = path_creator(location, layout, exp, "Vel", "Ar", file_name)
        AR_graph(outputPath, patterns[layout], AR_Vel, vline)

        n += 2


def NA(acc):
    max_acc = array([max(abs(i)) for i in acc])
    max_acc = sorted(max_acc, reverse=True)
    NA_acc = max_acc / max_acc[0]
    return NA_acc


"""outputFolder = "D:\Projeler\Tubitak\Output"
file1 = f"Milas_OT_L2_40Hz_2D_Pressure"
file2 = f"Milas_OT_L2_40Hz_3D_Pressure_Inf"
AccFileName1 = os.path.join(outputFolder,file1,f"{file1}_Acc.txt")
AccFileName2 = os.path.join(outputFolder,file2,f"{file2}_Acc.txt")
acc1 = read_file(AccFileName1)
NA1 = NA(acc1)[:9]
acc2 = read_file(AccFileName2)
NA2 = NA(acc2)[:9]
NA3 = fieldResults("OT-L25","40(40)")
x = [1,2,3.5,6,8.5,11,13.5,16,18.5]
plt.plot(x,NA1,label="2D")
plt.plot(x,NA2,label="3D")
plt.plot(x,NA3,label="Field")
plt.grid()
plt.legend()
plt.semilogy()
plt.title("OT-L25(40Hz)")
plt.show()"""

exp_list = ["A", "OT", "RC", "SP", "SP-OT", "SP-RC", "RC-SP", "OT-SP"]
layouts = ["L1", "L2", "L3", "L4", "L5"]
locations = ["Mentese", "Milas", "Senaryo1", "Senaryo2"]
frequencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]


total = (len(exp_list) * len(layouts) * len(frequencies) * len(locations) * 4) + (
    (len(exp_list) - 1) * len(layouts) * len(frequencies) * len(locations) * 2
)

for city in locations:
    for fr in frequencies:
        for layout in layouts:
            for exp in exp_list:
                try:
                    drawGraph(exp, layout, fr, city)
                except Exception as e:
                    print(e)
                    pass
                print(f"{n}\{total}")
