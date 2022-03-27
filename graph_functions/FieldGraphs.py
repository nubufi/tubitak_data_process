from subroutines.Time_Series import fromAcc, fromVel
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas import read_excel

exp_list = ["A", "OT", "RC", "SP", "SP-OT", "SP-RC"]  # Source sağdaymış gibi
layouts = ["L25", "L50"]
frequencies = np.arange(10, 160, 10)
locations = ["Menteşe", "Milas"]
distances = {
    "L25": [1, 2, 3.5, 6, 8.5, 11, 13.5, 16],
    "L50": [1, 2.5, 6, 8.5, 11, 13.5, 16, 18.5],
}
vlines = {"L25": 4.75 + 0.75 / 2, "L50": 7.25 + 0.75 / 2}


def run(func):
    arg_list = [
        ("A",),
        ("OT",),
        ("RC",),
        ("SP",),
        ("SP-OT",),
        ("SP-RC",),
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


def path_creator(path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, f"{file_name}.png")


def readFile(dataType, city, layout, exp, frequency, mainFolder="Filtered&Trimmed"):
    path = f"F:\\DataFolder\\{mainFolder}\\{dataType}\\1 - Time History\\{dataType} Time History\\{city}\\{exp}-{layout}"
    if mainFolder == "Filtered&Trimmed":
        fileName = f"{frequency}Hz.txt"
    else:
        fileName = f"{exp}-{frequency}-{layout}_{dataType[:3]}.txt"

    datas = np.loadtxt(os.path.join(path, fileName), unpack=True)

    return datas


def readFourier(dataType, city, layout, exp, frequency, mainFolder="Filtered&Trimmed"):
    path = f"F:\\DataFolder\\{mainFolder}\\{dataType}\\1 - Time History\\Fourier Spectrum\\{city}\\{exp}-{layout}"
    if mainFolder == "Filtered&Trimmed":
        fileName = f"{frequency}Hz.txt"
    else:
        fileName = f"{exp}-{frequency}-{layout}_{dataType[:3]}.txt"

    datas = np.loadtxt(os.path.join(path, fileName), unpack=True)

    return datas


def readNA(dataType, location, pattern, exp, frequency):
    path = f"F:\\DataFolder\\Filtered&Trimmed\\{dataType}\\3 - Normalized Values\\{location}\\Normalized {dataType} - S1.xlsx"
    excel = read_excel(path, sheet_name=f"{exp}-{pattern}")
    F_col = excel.iloc[:, 0]
    index = np.where(F_col == frequency)[0][0]

    return excel.iloc[index, 1:-1].values


def readAR(dataType, location, pattern, exp, frequency):
    path = f"F:\\DataFolder\\Filtered&Trimmed\\{dataType}\\5 - AR(Direct Method)\\{location}\\{dataType} (1, 2, 3 and Average).xlsx"
    excel = read_excel(path, sheet_name=f"{exp}-{pattern}")
    F_col = excel.iloc[:, 0]
    index = np.where(F_col == frequency)[0][0]

    return excel.iloc[index, [4, 8, 12, 16, 20]].values


# -------------------4.1----------------------------------------------------------
# Acc&Vel Time History
def timeHistory(city, y_label, dataType, layout):
    dt = 1 / 2000
    for frequency in frequencies:
        for exp in exp_list:
            data = readFile(dataType, city, layout, exp, frequency)
            folder = f"D:\\Projeler\\Tubitak\\Rapor Grafikleri\\Time History\\{dataType}\\{city}\\{exp}-{layout}\\{frequency}"
            print(f"{dataType}\\{city}\\{exp}-{layout}\\{frequency}")
            for i, d in enumerate(data):
                time = np.arange(0, len(d) * dt, dt)
                plt.plot(time, d)
                plt.ylabel(y_label)
                plt.xlabel("Zaman (s)")
                plt.grid()
                plt.xlim(0, len(d) * dt)
                path = path_creator(folder, f"S{i+1}")
                plt.savefig(path, bbox_inches="tight")
                plt.close()


# ivme ölçer vs jeofon time history
def doubleTimeHistory(city, y_label, targetType, layout):
    dt = 1 / 2000
    for frequency in frequencies:
        for exp in exp_list:
            acc = readFile("Acceleration", city, layout, exp, frequency)
            vel = readFile("Velocity", city, layout, exp, frequency)
            original = acc if targetType == "Acceleration" else vel
            converted = fromVel(vel) if targetType == "Acceleration" else fromAcc(acc)
            org_label = "İvme Sensörü" if targetType == "Acceleration" else "Jeofon"
            conv_label = "Jeofon" if targetType == "Acceleration" else "İvme Sensörü"
            folder = f"D:\\Projeler\\Tubitak\\Rapor Grafikleri\\Time History(Comparison)\\{targetType}\\{city}\\{exp}-{layout}\\{frequency}"
            print(f"{targetType}\\{city}\\{exp}-{layout}\\{frequency}")
            for i, d in enumerate(original):
                time = np.arange(0, len(d) * dt, dt)
                plt.plot(time, original[i], label=org_label)
                plt.plot(time, converted[i], label=conv_label)
                plt.ylabel(y_label)
                plt.xlabel("Zaman (s)")
                plt.legend(bbox_to_anchor=(1.01, 0.5), loc="upper left", frameon=False)
                plt.grid()
                plt.xlim(0, len(d) * dt)
                path = path_creator(folder, f"S{i+1}")
                plt.savefig(path, bbox_inches="tight")
                plt.close()


# NA vs NV
def NAComparison(exp):
    for L in layouts:
        for F in frequencies:
            for C in locations:
                folder = f"D:\\Projeler\\Tubitak\\Rapor Grafikleri\\NA(Comparison)\\{C}\\{exp}-{L}"
                print(f"{C}\\{exp}-{L}")
                path = path_creator(folder, f"{F}Hz")
                NA = readNA("Acceleration", C, L, exp, F)
                NV = readNA("Velocity", C, L, exp, F)
                distance = distances[L]
                plt.plot(distance, NA, marker="s", label="Normalize ivme")
                plt.plot(distance, NV, marker="o", label="Normalize hız")
                plt.semilogy()
                plt.yticks(
                    [0.0001, 0.001, 0.01, 0.1, 1, 10], [0.0001, 0.001, 0.01, 0.1, 1, 10]
                )
                plt.xticks([0, 5, 10, 15, 20])
                plt.grid()
                plt.ylabel("Normalize genlik")
                plt.xlabel("Mesafe (m)")
                plt.xlim((0, 20))
                plt.ylim(0.0001, 10)
                if exp != "A":
                    plt.vlines(vlines[L], 0.0001, 10, label="Dalga bariyeri")
                plt.legend(bbox_to_anchor=(1.01, 0.5), loc="upper left", frameon=False)
                plt.xticks(np.arange(0, 22.5, 2.5))
                plt.savefig(path, bbox_inches="tight")
                plt.close()


run(NAComparison)
# Fourier
def Fourier(city, layout, dataType):
    for exp in exp_list:
        for F in frequencies:
            data = readFourier(dataType, city, layout, exp, F)
            f = data[0]
            fa = data[1:]
            folder = f"D:\\Projeler\\Tubitak\\Rapor Grafikleri\\Fourier Spectrum\\{dataType}\\{city}\\{exp}-{layout}\\{F}"
            print(f"{dataType}\\{city}\\{exp}-{layout}\\{F}")
            for i, d in enumerate(fa):
                plt.plot(f, d)
                plt.ylabel("Spektral genlik")
                plt.xlabel("Frekans (Hz)")
                plt.grid()
                plt.xlim(0, 200)
                plt.ylim(0, max(d) * 1.05)
                path = path_creator(folder, f"S{i+1}")
                plt.savefig(path, bbox_inches="tight")
                plt.close()


# -------------------4.2----------------------------------------------------------
# NA&NV -> Menteşe A-L2, A-L3 , Milas A-L2,A-L3 (a)10-50 b)60-100 c)110-150)
def NACombined(city, layout, dataType):
    freqs = [np.arange(10, 60, 10), np.arange(60, 110, 10), np.arange(110, 160, 10)]
    folder = f"D:\\Projeler\\Tubitak\\Rapor Grafikleri\\NACombined\\{dataType}\\{city}\\{layout}"
    distance = distances[layout]
    labels = {"Acceleration": "Normalize ivme", "Velocity": "Normalize hız"}

    for domain in freqs:
        path = path_creator(folder, f"{domain[0]}-{domain[-1]}")
        print(path)
        for f in domain:
            NA = readNA(dataType, city, layout, "A", f)
            plt.plot(distance, NA, label=f"{f}Hz")
            plt.semilogy()
        plt.yticks([0.0001, 0.001, 0.01, 0.1, 1, 10], [0.0001, 0.001, 0.01, 0.1, 1, 10])
        plt.xticks([0, 5, 10, 15, 20])
        plt.grid()
        plt.legend(bbox_to_anchor=(1.01, 0.5), loc="upper left", frameon=False)
        plt.ylabel(labels[dataType])
        plt.xlabel("Mesafe (m)")
        plt.xlim((0, 20))
        plt.xticks(np.arange(0, 22.5, 2.5))
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# NA&NV A vs Single Barriers
# NA&NV A vs Coupled Barriers
def NABarriers(city, layout, dataType):
    distance = distances[layout]
    labels = {"Acceleration": "Normalize ivme", "Velocity": "Normalize hız"}
    barrierLabels = {
        "OT": "İçi boş hendek",
        "SP": "Palplanş",
        "RC": "Kauçuk dolu hendek",
        "SP-OT": "Boş hendek + Palplanş",
        "SP-RC": "Kauçuk dolu hendek + Palplanş",
    }
    for barrierType in ["Single", "Coupled"]:
        barriers = ["OT", "SP", "RC"] if barrierType == "Single" else ["SP-OT", "SP-RC"]
        for frequency in frequencies:
            print(barrierType, frequency, city, dataType, layout)
            folder = f"D:\\Projeler\\Tubitak\\Rapor Grafikleri\\NABarriers\\{dataType}\\{city}\\{layout}\\{frequency}"
            path = path_creator(folder, barrierType)
            A = readNA(dataType, city, layout, "A", frequency)
            plt.plot(distance, A, label="Bariyersiz durum")
            for exp in barriers:
                B = readNA(dataType, city, layout, exp, frequency)
                plt.plot(distance, B, label=barrierLabels[exp])
            plt.semilogy()
            plt.yticks(
                [0.0001, 0.001, 0.01, 0.1, 1, 10], [0.0001, 0.001, 0.01, 0.1, 1, 10]
            )
            plt.xticks([0, 5, 10, 15, 20])
            plt.grid()
            plt.ylabel(labels[dataType])
            plt.xlabel("Mesafe (m)")
            plt.xlim((0, 20))
            plt.ylim(0.0001, 10)
            plt.vlines(vlines[layout], 0.0001, 10, label="Dalga bariyeri")
            plt.legend(bbox_to_anchor=(1.01, 0.5), loc="upper left", frameon=False)
            plt.xticks(np.arange(0, 22.5, 2.5))
            plt.savefig(path, bbox_inches="tight")
            plt.close()


# -------------------4.3----------------------------------------------------------
# AR_direct(1,2,3 ortalaması) vs Distance (bariyer sonrası)
def AR_distance(city, layout, dataType):
    distance = distances[layout][3:]

    for exp in exp_list[1:]:
        folder = f"D:\\Projeler\\Tubitak\\Rapor Grafikleri\\AR_Distance\\{dataType}\\{city}\\{exp}-{layout}"
        for F in frequencies:
            path = path_creator(folder, f"{F}Hz")
            print(path)
            AR = readAR(dataType, city, layout, exp, F)
            plt.scatter(distance, AR)
            plt.grid()
            plt.vlines(vlines[layout], 0, 2, label="Dalga bariyeri")
            plt.legend(bbox_to_anchor=(1.01, 0.5), loc="upper left", frameon=False)
            plt.ylabel("Titreşim azalım oranı")
            plt.xlabel("Mesafe (m)")
            plt.xlim((0, 20))
            plt.ylim(0, 2)
            plt.yticks(np.arange(0, 2.25, 0.25))
            plt.xticks(np.arange(0, 22.5, 2.5))
            plt.savefig(path, bbox_inches="tight")
            plt.close()


# AR vs Distance - Single barriers
# AR vs Distance - Coupled barriers
def ARBarriers(city, layout, dataType):
    distance = distances[layout][3:]
    barrierLabels = {
        "OT": "İçi boş hendek",
        "SP": "Palplanş",
        "RC": "Kauçuk dolu hendek",
        "SP-OT": "Boş hendek + Palplanş",
        "SP-RC": "Kauçuk dolu hendek + Palplanş",
    }
    for barrierType in ["Single", "Coupled"]:
        barriers = ["OT", "SP", "RC"] if barrierType == "Single" else ["SP-OT", "SP-RC"]
        for frequency in frequencies:
            print(barrierType, frequency, city, dataType, layout)
            folder = f"D:\\Projeler\\Tubitak\\Rapor Grafikleri\\ARBarriers\\{dataType}\\{city}\\{layout}\\{frequency}"
            path = path_creator(folder, barrierType)
            for exp in barriers:
                B = readAR(dataType, city, layout, exp, frequency)
                plt.scatter(distance, B, label=barrierLabels[exp])
            plt.grid()
            plt.vlines(vlines[layout], 0, 2, label="Dalga bariyeri")
            plt.legend(bbox_to_anchor=(1.01, 0.5), loc="upper left", frameon=False)
            plt.ylabel("Titreşim azalım oranı")
            plt.xlabel("Mesafe (m)")
            plt.xlim((0, 20))
            plt.ylim(0, 2)
            plt.yticks(np.arange(0, 2.25, 0.25))
            plt.xticks(np.arange(0, 22.5, 2.5))
            plt.savefig(path, bbox_inches="tight")
            plt.close()
