import matplotlib.pyplot as plt
import numpy as np
import os
from pandas import read_excel,DataFrame
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
exp_list = ["A","OT","RC","SP","SP-OT","SP-RC","RC-SP","OT-SP"] #Source sağdaymış gibi
layouts = ["L1","L2","L3","L4","L5"]
frequencies = np.arange(10,160,10)
locations = ["Mentese","Milas","Senaryo1","Senaryo3","Senaryo2"]
distances = {
    "L1":[1,1.5,2,3.5,6,8.5,11,13.5,16,18.5,21,25,29],
    "L2":[1,2,3.5,6,8.5,11,13.5,16,18.5,21,25,29],
    "L3":[1,2.5,6,8.5,11,13.5,16,18.5,21,25,29],
    "L4":[1,4.75,8.75,11,13.5,16,18.5,21,25,29],
    "L5":[1,6,8.5,11,13.5,16,18.5,25,29],
}
vlines = {
    "L1":2.25+0.75/2,
    "L2":4.75+0.75/2,
    "L3":7.25+0.75/2,
    "L4":9.75+0.75/2,
    "L5":12.25+0.75/2,
}
VR_list = {"Milas":260.6,"Mentese":230,"Senaryo1":90,"Senaryo2":450,"Senaryo3":270}

def path_creator(path,file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    
    return os.path.join(path,f"{file_name}.png")

def readNA(dataType,location,pattern,exp,frequency):
    path = f"D:\Projects\Tubitak\Output\Excel\{dataType}\\3 - Normalized Values\{location}\\{pattern}.xlsx"
    excel = read_excel(path,sheet_name=exp)
    F_col = excel.iloc[:,0]
    index = np.where(F_col==frequency)[0][0]
    
    return excel.iloc[index,1:].values

def readAR(dataType,location,pattern,exp,frequency):
    path = f"D:\Projects\Tubitak\Output\Excel\{dataType}\\5 - AR(Direct Method)\\{location}\\{pattern}.xlsx"
    excel = read_excel(path,sheet_name=f"{exp}")
    F_col = excel.iloc[:,0]
    index = np.where(F_col==frequency)[0][0]
    firstCols = 4 if pattern == "L5" else 3 
    cols = np.arange(4,(len(distances[pattern])-firstCols)*4+4,4)
    return excel.iloc[index,cols].values

def readARAvg(dataType,location,pattern,exp,region,method="Numeric"):
    if method == "Field":
        dT = "Acceleration" if dataType == "Acc" else "Velocity"
        path = f"G:\DataFolder\Filtered&Trimmed\{dT}\\6 - AR Direct (Avg)\\{location}-{region}.xlsx"
    else:
        path = f"D:\Projects\Tubitak\Output\Excel\{dataType}\\6 - AR Direct (Avg)\\{location}-{region}.xlsx"
    excel = read_excel(path,sheet_name=f"{exp}")
    F_col = excel.iloc[:,0]
    index = np.where(F_col==pattern)[0][0]

    return excel.iloc[index,1:].values

def readFieldNA(dataType,location,pattern,exp,frequency):
    dataType = "Acceleration" if dataType == "Acc" else "Velocity"
    location = "Menteşe" if location == "Mentese" else location
    pattern = "L25" if pattern == "L2" else "L50"
    path = f"G:\\DataFolder\\Filtered&Trimmed\\{dataType}\\3 - Normalized Values\\{location}\\Normalized {dataType} - S1.xlsx"
    excel = read_excel(path,sheet_name=f"{exp}-{pattern}")
    F_col = excel.iloc[:,0]
    index = np.where(F_col==frequency)[0][0]
    
    return excel.iloc[index,1:-1].values

def readFieldAR(dataType,location,pattern,exp,frequency):
    dataType = "Acceleration" if dataType == "Acc" else "Velocity"
    location = "Menteşe" if location == "Mentese" else location
    pattern = "L25" if pattern == "L2" else "L50"
    path = f"G:\DataFolder\Filtered&Trimmed\{dataType}\\5 - AR(Direct Method)\{location}\\{dataType} (1, 2, 3 and Average).xlsx"
    excel = read_excel(path,sheet_name=f"{exp}-{pattern}")
    F_col = excel.iloc[:,0]
    index = np.where(F_col==frequency)[0][0]
    
    return excel.iloc[index,[4,8,12,16,20]].values

def regression(x,y,order=3):
    cons = np.polyfit(np.array(x,dtype=float),np.array(y,dtype=float),order)
    f = np.poly1d(cons)
    return cons,f(frequencies)

def regression_exp(x,y):
    cons = np.polyfit(np.array(x,dtype=float),np.log(np.array(y,dtype=float)),1)
    f = np.poly1d(cons)
    pred = np.exp(cons[1])*np.exp(cons[0]*frequencies)
    return pred

def run(func):
    arg_list = [
        ("A",),
        ("OT",),
        ("RC",),
        ("SP",),
        ("SP-OT",),
        ("SP-RC",),
        ("RC-SP",),
        ("OT-SP",),
    ]

    processDict = {}
    import multiprocessing
    if __name__ == '__main__':
        for i,arg in enumerate(arg_list):
            processDict[i] = multiprocessing.Process(target=func,args=arg)

        for p in processDict:
            processDict[p].start()
        
        for p in processDict:
            processDict[p].join()

#NAvsNV
def NAComparison(exp):
    for L in layouts:
        for F in frequencies:
            for C in locations:
                folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\NA(Comparison)\\{C}\\{exp}\\{L}"
                path = path_creator(folder,f"{F}Hz")
                NA = readNA("Acc",C,L,exp,F)
                NV = readNA("Vel",C,L,exp,F)
                distance = distances[L]
                plt.plot(distance,NA,marker="s",label="Normalize ivme")
                plt.plot(distance,NV,marker="o",label="Normalize hız")
                plt.semilogy()
                plt.yticks([0.0001,0.001,0.01,0.1,1,10],[0.0001,0.001,0.01,0.1,1,10])
                plt.grid()
                plt.ylabel("Normalize genlik")
                plt.xlabel("Mesafe (m)")
                plt.xlim((0,30))
                plt.ylim(0.0001,10)
                if exp!="A":
                    plt.vlines(vlines[L],0.0001,10,label="Dalga bariyeri")
                plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                plt.xticks(np.arange(0,32.5,2.5))
                
                plt.savefig(path,bbox_inches='tight')
                plt.close()

#Field vs Numeric NA
def FieldvsNumericNA(exp):
    y_labels = {"Acc":"Normalize ivme","Vel":"Normalize hız"}
    exceptions = ["OT-SP","RC-SP"]
    if exp not in exceptions:
        for L in ["L2","L3"]:
            for F in frequencies:
                for C in ["Mentese","Milas"]:
                    for dataType in ["Acc","Vel"]:
                        folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\FieldvsNumeric\\NA\\{dataType}\\{C}\\{exp}\\{L}"
                        path = path_creator(folder,f"{F}Hz")
                        numeric = readNA(dataType,C,L,exp,F)
                        field = readFieldNA(dataType,C,L,exp,F)
                        distance = distances[L]
                        plt.plot(distance,numeric,label="Sayısal model")
                        plt.scatter(distance[:len(field)],field,label="Saha deneyi",color="Orange")
                        plt.semilogy()
                        plt.yticks([0.0001,0.001,0.01,0.1,1,10],[0.0001,0.001,0.01,0.1,1,10])
                        plt.grid()
                        plt.ylabel(y_labels[dataType])
                        plt.xlabel("Mesafe (m)")
                        plt.xlim((0,30))
                        plt.ylim(0.0001,10)
                        if exp!="A":
                            plt.vlines(vlines[L],0.0001,10,label="Dalga bariyeri")
                        plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                        plt.xticks(np.arange(0,32.5,2.5))
                        plt.savefig(path,bbox_inches='tight')
                        plt.close()

def FieldvsNumericAR(exp):
    exceptions = ["OT-SP","RC-SP","A"]
    if exp not in exceptions:
        for L in ["L2","L3"]:
            for F in frequencies:
                for C in ["Mentese","Milas"]:
                    for dataType in ["Acc","Vel"]:
                        folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\FieldvsNumeric\\AR\\{dataType}\\{C}\\{exp}\\{L}"
                        path = path_creator(folder,f"{F}Hz")
                        numeric = readAR(dataType,C,L,exp,F)
                        field = readFieldAR(dataType,C,L,exp,F)
                        distance = distances[L][3:]
                        plt.plot(distance,numeric,label="Sayısal model")
                        plt.scatter(distance[:len(field)],field,label="Saha deneyi",color="Orange")
                        plt.grid()
                        plt.vlines(vlines[L],0,2,label="Dalga bariyeri")
                        plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                        plt.ylabel("Titreşim azalım oranı")
                        plt.xlabel("Mesafe (m)")
                        plt.xlim((0,30))
                        plt.ylim(0,2)
                        plt.yticks(np.arange(0,2.25,0.25))
                        plt.xticks(np.arange(0,32.5,2.5))
                        plt.savefig(path,bbox_inches='tight')
                        plt.close()

#NA&NV
def NA(exp):
    y_labels = {"Acc":"Normalize ivme","Vel":"Normalize hız"}
    for L in layouts:
        for F in frequencies:
            for C in locations:
                for dataType in ["Acc","Vel"]:
                    folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\NormalizedAmplitudes\\{dataType}\\{C}\\{exp}\\{L}"
                    path = path_creator(folder,f"{F}Hz")
                    NA = readNA(dataType,C,L,exp,F)
                    distance = distances[L]
                    plt.plot(distance,NA,marker="s")
                    plt.semilogy()
                    plt.yticks([0.0001,0.001,0.01,0.1,1,10],[0.0001,0.001,0.01,0.1,1,10])
                    plt.grid()
                    plt.ylabel(y_labels[dataType])
                    plt.xlabel("Mesafe (m)")
                    plt.xlim((0,30))
                    plt.ylim(0.0001,10)
                    if exp!="A":
                        plt.vlines(vlines[L],0.0001,10,label="Dalga bariyeri")
                    plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                    plt.xticks(np.arange(0,32.5,2.5))
                    plt.savefig(path,bbox_inches='tight')
                    plt.close()

#AR-Distance
def AR_distance(exp):
    if exp!="A":
        for L in layouts:
            distance = distances[L][4:] if L == "L5" else distances[L][3:]
            for F in frequencies:
                for C in locations:
                    for dataType in ["Acc","Vel"]:
                        folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\AR_Distance\\{dataType}\\{C}\\{exp}\\{L}"
                        path = path_creator(folder,f"{F}Hz")
                        AR = readAR(dataType,C,L,exp,F)
                        plt.scatter(distance,AR)
                        plt.grid()
                        plt.vlines(vlines[L],0,2,label="Dalga bariyeri")
                        plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                        plt.ylabel("Titreşim azalım oranı")
                        plt.xlabel("Mesafe (m)")
                        plt.xlim((0,30))
                        plt.ylim(0,2)
                        plt.yticks(np.arange(0,2.25,0.25))
                        plt.xticks(np.arange(0,32.5,2.5))
                        plt.savefig(path,bbox_inches='tight')
                        plt.close()

#NA&NV (10-50) (60-100) (110-150)
def NACombined(exp):
    freqs = [np.arange(10,60,10),np.arange(60,110,10),np.arange(110,160,10)]
    y_labels = {"Acc":"Normalize ivme","Vel":"Normalize hız"}
    markers = ["s","o","v","^","*"]
    for L in layouts:
        distance = distances[L]
        for C in locations:
            for dataType in ["Acc","Vel"]:
                folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\NACombined\\{dataType}\\{C}\\{exp}\\{L}"
                for domain in freqs:
                    path = path_creator(folder,f"{domain[0]}-{domain[-1]}")
                    for i,f in enumerate(domain):
                        NA = readNA(dataType,C,L,exp,f)
                        plt.plot(distance,NA,marker=markers[i],label=f"{f}Hz")
                    plt.semilogy()
                    plt.yticks([0.0001,0.001,0.01,0.1,1,10],[0.0001,0.001,0.01,0.1,1,10])
                    plt.grid()
                    plt.ylabel(y_labels[dataType])
                    plt.xlabel("Mesafe (m)")
                    plt.xlim((0,30))
                    plt.ylim(0.0001,10)
                    if exp!="A":
                        plt.vlines(vlines[L],0.0001,10,label="Dalga bariyeri")
                    plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                    plt.xticks(np.arange(0,32.5,2.5))
                    plt.savefig(path,bbox_inches='tight')
                    plt.close()

#AR (10-50) (60-100) (110-150)
def ARCombined(exp):
    freqs = [np.arange(10,60,10),np.arange(60,110,10),np.arange(110,160,10)]
    markers = ["o","D","s","^","x"]
    colors = ["black","darkblue","grey","green","crimson"]
    if exp!="A":
        for L in layouts:
            distance = distances[L][4:] if L == "L5" else distances[L][3:]
            for C in locations:
                for dataType in ["Acc","Vel"]:
                    folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\ARCombined\\{dataType}\\{C}\\{exp}\\{L}"
                    for domain in freqs:
                        path = path_creator(folder,f"{domain[0]}-{domain[-1]}")
                        for i,f in enumerate(domain):
                            AR = readAR(dataType,C,L,exp,f)
                            plt.plot(distance,AR,marker=markers[i],color=colors[i],label=f"{f}Hz")
                        plt.grid()
                        plt.vlines(vlines[L],0,2,label="Dalga bariyeri")
                        plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                        plt.ylabel("Titreşim azalım oranı")
                        plt.xlabel("Mesafe (m)")
                        plt.xlim((0,30))
                        plt.ylim(0,2)
                        plt.yticks(np.arange(0,2.25,0.25))
                        plt.xticks(np.arange(0,32.5,2.5))
                        plt.savefig(path,bbox_inches='tight')
                        plt.close()

#AR-Frequency
def AR_frequency(exp):
    field_patterns = ["L2","L3"]
    field_locations = ["Mentese","Milas"]
    field_exps = ["OT","RC","SP","SP-OT","SP-RC"]
    fieldLabels = {"Mentese":"Menteşe","Milas":"Milas","Senaryo1":"Vs=100m/s","Senaryo2":"Vs=500m/s","Senaryo3":"Vs=300m/s"}
    colors = {"Milas":"crimson","Mentese":"green","Senaryo1":"grey","Senaryo3":"darkblue","Senaryo2":"black"}
    symbols = {"Milas":"o","Mentese":"D","Senaryo1":"x","Senaryo3":"^","Senaryo2":"s"}
    if exp != "A":
        for L in layouts:
            for dataType in ["Acc","Vel"]:
                for region in ["far","near"]:
                    all_ = []
                    folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\AR Direct (Avg)\\{dataType}\\{exp}"
                    path = path_creator(folder,f"{L}-{region}")
                    fields = []
                    numerics = []
                    for C in locations:
                        numeric = readARAvg(dataType,C,L,exp,region)
                        all_.append(numeric)
                        
                        if L in field_patterns and C in field_locations and exp in field_exps:
                            field = readARAvg(dataType,C,L,exp,region,method="Field")
                            p,= plt.plot(frequencies[:8],field[:8],linewidth=0,label=f"{fieldLabels[C]}(Saha deneyi)",color=colors[C],marker=symbols[C])
                            fields.append(p)
                        p,=plt.plot(frequencies,numeric,label=f"{fieldLabels[C]}(Sayısal)",color=colors[C],marker=symbols[C])
                        numerics.append(p)

                    plt.grid()
                    plt.ylabel("Titreşim azalım oranı")
                    plt.xlabel("Frekans (Hz)")
                    plt.xlim((0,155))
                    plt.ylim(0,2)
                    plt.yticks(np.arange(0,2.25,0.25))
                    plt.xticks(frequencies)
                    L1 = plt.legend(handles=fields,bbox_to_anchor=(1.01, 0.6),loc='upper left',frameon=False)
                    L2 = plt.legend(handles=numerics,bbox_to_anchor=(1.01, 0.47),loc='upper left',frameon=False)
                    plt.gca().add_artist(L1)
                    plt.gca().add_artist(L2)
                    #plt.show()
                    plt.savefig(path,bbox_inches='tight')
                    plt.close()

def readSP_AR(dataType,location,pattern,depth,region):
    path = f"D:\Projects\Tubitak\Output\Excel\SP\{dataType}\\6 - AR Direct (Avg)\\{location}-{region}.xlsx"

    excel = read_excel(path,sheet_name=f"{depth}m")
    F_col = excel.iloc[:,0]
    index = np.where(F_col==pattern)[0][0]

    return excel.iloc[index,1:].values

def SP_AR_regression(dataType,pattern,depth,region):
    temp = np.zeros(15)
    for location in locations:
        AR = np.array(readSP_AR(dataType,location,pattern,depth,region),dtype=float)
        temp = temp + AR
    avg = temp/5
    const = np.polyfit(np.array(frequencies,dtype=float),avg,3)
    f = np.poly1d(const)
    return f(frequencies)

def SPGraph():
    markers = ["o","D","s","^","x"]
    colors = ["black","darkblue","grey","green","crimson"]
    for dataType in ["Acc","Vel"]:
        folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\SP\\AR Direct (Avg)\\{dataType}"
        for pattern in layouts:
            for region in ["near","far"]:
                path = path_creator(folder,f"{pattern}-{region}")
                for i,depth in enumerate([4,6,8,12,18]):
                    AR = SP_AR_regression(dataType,pattern,depth,region)
                    plt.plot(frequencies,AR,label=f"{depth}m",color=colors[i],marker=markers[i])
                plt.grid()
                plt.ylabel("Titreşim azalım oranı")
                plt.xlabel("Frekans (Hz)")
                plt.xlim((0,155))
                plt.ylim(0,2)
                plt.yticks(np.arange(0,2.25,0.25))
                plt.xticks(frequencies)
                plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                #plt.show()
                plt.savefig(path,bbox_inches='tight')
                plt.close()


def coupleGraph(exp):
    markers = ["^","o","D","x"]
    colors = ["crimson","blue","black","green"]
    labels = {
        "OT":"İçi boş hendek",
        "RC":"Kauçuk dolu hendek",
        "SP-OT":"İçi boş hendek + Palplanş",
        "SP-RC":"Kauçuk dolu hendek + Palplanş",
        "OT-SP":"Palplanş + İçi boş hendek",
        "RC-SP":"Palplanş + Kauçuk dolu hendek"}
    for dataType in ["Acc","Vel"]:
        folder = f"D:\\Projects\\Tubitak\\Rapor Grafikleri\\Numeric\\AR Direct (Coupled - {exp})\\{dataType}"
        for pattern in layouts:
            for region in ["near","far"]:
                path = path_creator(folder,f"{pattern}-{region}")
                SP = AR_regression(dataType,pattern,"SP",region)
                B = AR_regression(dataType,pattern,exp,region)
                Couple = AR_regression(dataType,pattern,f"SP-{exp}",region)
                Couple_reverse = AR_regression(dataType,pattern,f"{exp}-SP",region)
                plt.plot(frequencies,SP,label="Palplanş",color=colors[0],marker=markers[0])
                plt.plot(frequencies,B,label=labels[exp],color=colors[1],marker=markers[1])
                plt.plot(frequencies,Couple,label=labels[f"SP-{exp}"],color=colors[2],marker=markers[2])
                plt.plot(frequencies,Couple_reverse,label=labels[f"{exp}-SP"],color=colors[3],marker=markers[3])
                plt.grid()
                plt.ylabel("Titreşim azalım oranı")
                plt.xlabel("Frekans (Hz)")
                plt.xlim((0,155))
                plt.ylim(0,2)
                plt.yticks(np.arange(0,2.25,0.25))
                plt.xticks(frequencies)
                plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                #plt.show()
                plt.savefig(path,bbox_inches='tight')
                plt.close()

def NA3D(dataType,city,exp,pattern,frequency):
    model_name = f"{city}_{exp}_{pattern}_{frequency}Hz_3D"
    path = f"D:\\Projects\\Tubitak\\Output\\3D\\{model_name}"
    data = np.loadtxt(os.path.join(path,f"{model_name}_{dataType}.txt"),skiprows=1,unpack=True)[1:]
    MAX = sorted(np.max(abs(data),axis=1),reverse=True)
    return MAX/MAX[0]

def fullComparison():
    folder = "D:\Projects\Tubitak\Rapor Grafikleri\Selected\\5\\5.1\\1"
    for exp in ["A","OT","RC","SP","SP-OT","SP-RC"]:
        path = path_creator(folder,exp)
        D3 = NA3D("Acc","Milas",exp,"L3",50)
        D2 = readNA("Acc","Milas","L3",exp,50)
        Field = readFieldNA("Acc","Milas","L3",exp,50)
        plt.plot(distances["L3"][:len(Field)],Field,label="Saha deneyi")
        plt.plot(distances["L3"],D2,label="2D Sayısal analiz")
        plt.plot(distances["L3"],D3,label="3D Sayısal analiz")
        plt.semilogy()
        plt.yticks([0.0001,0.001,0.01,0.1,1,10],[0.0001,0.001,0.01,0.1,1,10])
        plt.grid()
        plt.ylabel("Normalize ivme")
        plt.xlabel("Mesafe (m)")
        plt.xlim((0,30))
        plt.ylim(0.0001,10)
        if exp!="A":
            plt.vlines(vlines["L3"],0.0001,10,label="Dalga bariyeri")
        plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
        plt.xticks(np.arange(0,32.5,2.5))
        plt.savefig(path,bbox_inches='tight')
        plt.close()

def calcDepth(exp,location):
    if exp in ["OT","RC"]:
        depth = 3
    else:
        depth = 8
    
    LR = VR_list[location]/frequencies
    D = depth/LR
    return D

def AR_regression_single(dataType,pattern,exp,region):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    temp_AR = []
    temp_D = []
    for location in locations:
        AR = readARAvg(dataType,location,pattern,exp,region)
        D = calcDepth(exp,location)
        temp_AR.extend(AR)
        temp_D.extend(D)

    temp_D = np.array(temp_D)
    temp_AR = np.array(temp_AR)
    indexes = np.where(temp_D<=1.8)[0]
    D = temp_D[indexes]
    AR = temp_AR[indexes]

    D_new = np.arange(0,max(temp_D)+0.01,0.01)
    popt, pcov = curve_fit(func, temp_D, temp_AR, p0=[1, 0.5, 1])
    y_pred = func(D_new,*popt)
    r2 = r2_score(temp_AR,func(temp_D,*popt))
    return *popt,D_new,y_pred,r2

def regressionGraphs_single():
    exps = ["OT","RC","SP"]
    colors = ["black","darkblue","crimson","green","grey"]
    for exp in exps:
        maxX = 14 if exp=="SP" else 5
        range_ = 1 if exp=="SP" else 0.5 
        for region in ["near","far"]:
            path = "D:\Projects\Tubitak\Output"
            for i,L in enumerate(layouts):
                a,b,c,D,AR,r2 = AR_regression_single("Acc",L,exp,region)
                plt.plot(D,AR,label=L,color=colors[i])
            plt.grid()
            plt.ylabel("Titreşim azalım oranı")
            plt.xlabel("Normalize bariyer derinliği")
            plt.legend()
            plt.xlim((0,maxX))
            plt.ylim(0,2)
            plt.yticks(np.arange(0,2.25,0.25))
            plt.xticks(np.arange(0,maxX+range_,range_))
            plt.savefig(os.path.join(path,f"{exp}-{region}"))
            plt.close()

def AR_regression_couple(dataType,pattern,exp,region):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    temp_AR = []
    temp_F = []
    for location in locations:
        AR = readARAvg(dataType,location,pattern,exp,region)
        temp_AR.extend(AR)
        temp_F.extend(frequencies)
        #plt.scatter(frequencies,AR)

    F_new = np.arange(0,151,1)
    temp_F = np.array(temp_F,dtype=float)
    popt, pcov = curve_fit(func, temp_F, temp_AR, p0=[0.1, 0.01, 0.1])
    y_pred = func(F_new,*popt)
    #plt.plot(F_new,y_pred)
    #plt.show()
    #plt.close()
    r2 = r2_score(temp_AR,func(temp_F,*popt))
    return *popt,F_new,y_pred,r2

def regressionGraphs_couple():
    exps = ["SP-OT","SP-RC","RC-SP","OT-SP"]
    colors = ["black","darkblue","crimson","green","grey"]
    exceptions = {
        "RC-SP-L1":[0.96,0.12,0.41],"RC-SP-L2":[0.93,0.13,0.35],
        "SP-OT-L1":[0.9,0.12,0.43],"SP-OT-L2":[0.85,0.1,0.28],"SP-RC-L1":[0.93,0.14,0.29],
        "SP-RC-L2":[0.88,0.11,0.4],"OT-SP-L1":[1.1,0.16,0.34],"OT-SP-L2":[0.95,0.15,0.37]}
    def func(a, b, c):
        x = np.arange(0,151,1)
        return a * np.exp(-b * x) + c
    for exp in exps:
        for region in ["far"]:
            path = "D:\Projects\Tubitak\Output"
            for i,L in enumerate(layouts):
                if f"{exp}-{L}" in exceptions.keys():
                    a,b,c = exceptions[f"{exp}-{L}"]
                    AR = func(a,b,c)
                else:
                    a,b,c,D,AR,r2 = AR_regression_couple("Acc",L,exp,region)
                plt.plot(np.arange(0,151,1),AR,label=L,color=colors[i])
            plt.grid()
            plt.ylabel("Titreşim azalım oranı")
            plt.xlabel("Frekans (Hz)")
            plt.legend()
            plt.xlim((0,150))
            plt.ylim(0,2)
            plt.yticks(np.arange(0,2.25,0.25))
            plt.xticks(np.arange(0,160,10))
            plt.savefig(os.path.join(path,f"{exp}-{region}"))
            plt.close()

def regressionExcel():
    cols = {"Deney":[],"Layout":[],"near":[],"far":[],"R2 Near":[],"R2 Far":[]}
    for exp in exp_list[1:]:
        x = "D" if exp in ["OT","SP","RC"] else "F"
        RFunc = AR_regression_single if exp in ["OT","SP","RC"] else AR_regression_couple
        for L in layouts:
            cols["Deney"].append(exp)
            cols["Layout"].append(L)
            for region in ["near","far"]:
                try:
                    a,b,c,D,AR,r2 = RFunc("Acc",L,exp,region)
                    eq = f"{a.round(2)}*e^({b.round(2)}{x})+{c.round(2)}"
                except:
                    eq = "-"
                    r2 = "-"
                cols[region].append(eq)
                if region == "near":
                    cols["R2 Near"].append(r2)
                else:
                    cols["R2 Far"].append(r2)
    DF = DataFrame.from_dict(cols)
    DF.to_excel("Denklemler.xlsx")
                
                
            
def readLiterature(exp,sheet):
    excel = read_excel(f"{exp}.xlsx",sheet_name=sheet)
    D = excel.iloc[:,0].values
    AR = excel.iloc[:,1].values
    return D,AR

def literatureData():
    Haupt = {
        "Data":readLiterature("OT","Haupt"),
        "Label":"Haupt(1981)-Hendeğin yakını"}
    Beskos = {
        "Data":readLiterature("OT","Beskos"),
        "Label":"Beskos vd. (1986)"
    }
    Alhussaini = {
        "Data":readLiterature("OT","Al-hossaini"),
        "Label":"Al-Hussaini (1992)"
    }
    Tsai = {
        "Data":readLiterature("OT","Tsai"),
        "Label":"Tsai ve Chang (2009)"
    }
    Alzawi = {
        "Data":readLiterature("OT","Alzawi"),
        "Label":"Alzawi ve El Naggar (2011)"
    }
    Saika = {
        "Data":readLiterature("OT","Saika"),
        "Label":"Saika ve Das (2014)"
    }
    Toygar = {
        "Data":readLiterature("OT","Toygar"),
        "Label":"Ülgen ve Toygar (2015)"
    }
    Mahdavisefat = {
        "Data":readLiterature("OT","Mahdavisefat"),
        "Label":"Mahdavisefat vd. (2017)"
    }
    Hassoun_OT = {
        "Data":readLiterature("OT","Hassoun"),
        "Label":"Hassoun (2018)"
    }
    Chew = {
        "Data":readLiterature("OT","Chew"),
        "Label":"Chew ve Leong (2019)"
    }
    Hassoun = {
        "Data":readLiterature("RC","Hassoun"),
        "Label":"Hassoun (2018)"
    }
    OT_list = [Haupt,Beskos,Alhussaini,Tsai,Alzawi,Saika,Toygar,Mahdavisefat,Hassoun_OT,Chew]
    RC_list = [Hassoun]

    return OT_list,RC_list
    
    
def literatureGraph():
    colors = ["black","darkblue","crimson","green","grey"]
    markers = ["o","v","^","<","1","s","p","P","*","D"]
    OT_list, RC_list = literatureData()
    exps = {"OT":OT_list,"RC":RC_list}
    for exp in exps.keys():
        maxX = 14 if exp=="SP" else 5
        range_ = 1 if exp=="SP" else 0.5
        Literature = exps[exp]
        for region in ["near"]:
            path = "D:\Projects\Tubitak\Output"
            for i,L in enumerate(layouts):
                a,b,c,D,AR,r2 = AR_regression_single("Acc",L,exp,region)
                plt.plot(D,AR,label=L,color=colors[i])
            for n,Lit in enumerate(Literature):
                data = Lit["Data"]
                label = Lit["Label"]
                plt.scatter(data[0],data[1],label=label,marker=markers[n])
            plt.grid()
            plt.ylabel("Titreşim azalım oranı")
            plt.xlabel("Normalize bariyer derinliği")
            plt.legend(bbox_to_anchor=(1.01, 1),loc='upper left',frameon=False)
            plt.xlim((0,maxX))
            plt.ylim(0,2)
            plt.yticks(np.arange(0,2.25,0.25))
            plt.xticks(np.arange(0,maxX+range_,range_))
            plt.savefig(os.path.join(path,f"{exp}"),bbox_inches='tight')
            plt.close()

regressionGraphs_couple()
#run(ARCombined)
#run(NAComparison)
#run(FieldvsNumericNA)
#run(FieldvsNumericAR)
#run(NA)
#run(AR_distance)
#run(NACombined)
#run(AR_frequency)
#coupleGraph("OT")
#coupleGraph("RC")
#fullComparison()
