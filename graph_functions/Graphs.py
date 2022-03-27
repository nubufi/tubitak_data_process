from numpy import loadtxt,array,where,mean,arange,append,flipud
import matplotlib.pyplot as plt
import os
from pandas import read_excel
import matplotlib
font = {'family' : 'normal',
        'fontname':'Times New Roman',
        'weight' : 'bold',
        'size'   : 11}
matplotlib.rc('font')
plt.rcParams['figure.figsize'] = 4.724, 3.149
patterns = {
    "Kamyon":{
        "L25":[0,1,2,3.5,6,8.5,11,13.5,16],
        "L50":[0,1,2.5,6,8.5,11,13.5,16,18.5]
    },
    "S1":{
        "L25":[1,2,3.5,6,8.5,11,13.5,16],
        "L50":[1,2.5,6,8.5,11,13.5,16,18.5]
    },
}
barrier = {"L25":5.15,"L50":7.625}
frequencies = arange(10,160,10)


def readNumeric(dataType,location,pattern,exp,frequency,fileName):
    path = f"D:\\Projeler\\Tubitak\\Datas\\Numeric\\Excel\\{dataType}\\{location}\\{pattern}\\{fileName}.xlsx"
    excel = read_excel(path,sheet_name=exp)
    F_col = excel.iloc[:,0]
    index = where(F_col==frequency)[0][0]
    
    return excel.iloc[index,1:-1].values

def readField(dataType,location,exp,frequency,fileName,sensor):
    path = f"D:\\Projeler\\Tubitak\\Datas\\Field\\Excel\\DirectMethod\\{dataType}\\{location}\\{fileName}.xlsx"
    excel = read_excel(path,sheet_name=exp)
    F_col = excel.iloc[:,0]
    index = where(F_col==frequency)[0][0]
    sensorIndexes = {
        1:[1,5,9,13,17],
        2:[2,6,10,14,18],
        3:[3,7,11,15,19],
        "avg":[4,8,12,16,20]
        }
    return excel.iloc[index,sensorIndexes[sensor]].values

def path_creator(path,file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    
    return os.path.join(path,f"{file_name}.png")

def objectCreator(color,marker,fillStyle,label,exp,dataType,frequency,city,sensor,lineType="-"):
    y = readField(dataType,city,exp,frequency,"AR",sensor)
    return {
        "color":color,
        "lineType":lineType,
        "marker":marker,
        "fillStyle":fillStyle,
        "label":label,
        "y":y}

#NA vs NV
def NAvsNV():
    for NA_type in ["Kamyon","S1"]:
        for pattern in ["L25","L50"]:
            for city in ["Milas","Menteşe"]:
                for frequency in frequencies:
                    path = path_creator("NAvsNV",city,NA_type,pattern,frequency)
                    NA = readData(city,"Accelerations","NA.xlsx",f"A-{pattern}",frequency,NA_type)
                    NV = readData(city,"Velocities","NA.xlsx",f"A-{pattern}",frequency,NA_type)
                    distance = patterns[NA_type][pattern]
                    plt.plot(distance,NA,marker="s",label="NA",color="Black")
                    plt.plot(distance,NV,marker="o",label="NV",color="Blue")
                    plt.semilogy()
                    plt.yticks([0.0001,0.001,0.01,0.1,1,10],[0.0001,0.001,0.01,0.1,1,10])
                    plt.xticks([0,5,10,15,20])
                    plt.grid()
                    plt.legend(bbox_to_anchor=(1.01, 0.5),loc='upper left',frameon=False)
                    plt.ylabel("Normalized amplitude")
                    plt.xlabel("Distance (m)")
                    plt.xlim((0,20))
                    plt.xticks(arange(0,22.5,2.5))
                    plt.savefig(path,dpi=1000,bbox_inches='tight')
                    plt.close()
    
def NormalisedAmplitude():
    y_labels = {
        "Accelerations":"Normalized Acceleration",
        "Velocities":"Normalized Velocity",
        "Disp(acc)":"Normalized Displacement",
        "Disp(vel)":"Normalized Displacement",
        "Fourier(acc)":"Normalized Spectral Amplitude",
        "Fourier(vel)":"Normalized Spectral Amplitude",
        }
    NA_type = "Kamyon"
    for pattern in ["L25","L50"]:
        distance = patterns[NA_type][pattern]
        for city in ["Milas","Menteşe"]:
            for dataType in ["Disp(vel)"]:
                for f in frequencies:
                    A = objectCreator("Black","_","full","Without barrier",f"A-{pattern}",dataType,NA_type,f,city,"NA.xlsx","--")
                    OT = objectCreator("Grey","o","none","Open trench",f"OT-{pattern}",dataType,NA_type,f,city,"NA.xlsx")
                    RC = objectCreator("Green","x","full","Rubber chips",f"RC-{pattern}",dataType,NA_type,f,city,"NA.xlsx")
                    SP = objectCreator("Blue","d","full","Sheet pile",f"SP-{pattern}",dataType,NA_type,f,city,"NA.xlsx")
                    OT_SP = objectCreator("Red","s","none","Open trench - Sheet pile",f"SP-OT-{pattern}",dataType,NA_type,f,city,"NA.xlsx")
                    RC_SP = objectCreator("Purple","^","full","Rubber chips - Sheet pile",f"SP-RC-{pattern}",dataType,NA_type,f,city,"NA.xlsx")
                    methods = {"Genel":[A,OT,RC,SP,OT_SP,RC_SP],
                            "Makale1":[A,OT,SP,OT_SP],
                            "Makale2":[A,RC,SP,RC_SP]
                            }
                    for method in methods:
                        print("NA",NA_type,pattern,city,dataType,method)
                        plots = methods[method]
                        for i in plots:
                            color = i["color"]
                            label = i["label"]
                            marker = i["marker"]
                            fillStyle = i["fillStyle"]
                            lineType = i["lineType"]
                            y = i["y"]
                            plt.plot(distance,y,color=color,marker=marker,label=label,fillstyle=fillStyle,linestyle=lineType)
                        plt.semilogy()
                        plt.yticks([0.0001,0.001,0.01,0.1,1,10],[0.0001,0.001,0.01,0.1,1,10])
                        plt.xticks([0,5,10,15,20])
                        plt.grid()
                        folder = f"D:\\Projeler\\Tubitak\\FinalGraphs\\NormalizedAmplitudes\\Baseline\\{method}\\{city}\\{dataType}\\{NA_type}\\{pattern}"
                        path = path_creator(folder,f)
                        plt.vlines(barrier[pattern],0.0001,10,label="Wave barrier",linewidth=2)
                        plt.legend(bbox_to_anchor=(1.01, 0.5),loc='center left',frameon=False)
                        plt.ylabel(y_labels[dataType])
                        plt.xlabel("Distance (m)")
                        plt.xlim((0,20))
                        plt.ylim((0.0001,10,))
                        plt.xticks(arange(0,22.5,2.5))
                        plt.savefig(path,dpi=1000,bbox_inches='tight')
                        #plt.show()
                        plt.close()

def AmplitudeReductionRatio(sensor):
    startFrom ={"Kamyon":4,"S1":3}
    NA_type = "S1"
    for pattern in ["L25","L50"]:
        distance = patterns[NA_type][pattern]
        for dataType in ["Accelerations","Velocities","Disp(acc)","Disp(vel)","Fourier(acc)","Fourier(vel)"]:
            for city in ["Milas","Menteşe"]:
                for f in frequencies:    
                    OT = objectCreator("Grey","o","none","Open trench",f"OT-{pattern}",dataType,f,city,sensor)
                    RC = objectCreator("Green","x","full","Rubber chips",f"RC-{pattern}",dataType,f,city,sensor)
                    SP = objectCreator("Blue","d","full","Sheet pile",f"SP-{pattern}",dataType,f,city,sensor)
                    OT_SP = objectCreator("Red","s","none","Open trench - Sheet pile",f"SP-OT-{pattern}",dataType,f,city,sensor)
                    RC_SP = objectCreator("Purple","^","full","Rubber chips - Sheet pile",f"SP-RC-{pattern}",dataType,f,city,sensor)
                    methods = {"Genel":[OT,RC,SP,OT_SP,RC_SP],
                            "Makale1":[OT,SP,OT_SP],
                            "Makale2":[RC,SP,RC_SP]
                            }
                    for method in methods:
                        print("AR",NA_type,pattern,city,dataType,method)
                        plots = methods[method]
                        for i in plots:
                            color = i["color"]
                            label = i["label"]
                            marker = i["marker"]
                            fillStyle = i["fillStyle"]
                            lineType = i["lineType"]
                            y = i["y"]
                            plt.plot(distance[startFrom[NA_type]:],y,color=color,marker=marker,label=label,fillstyle=fillStyle,linewidth=0)
                        plt.yticks(arange(0,2.25,0.25))
                        plt.grid()
                        folder = f"D:\\Projeler\\Tubitak\\FinalGraphs\\AR\\DirectMethod\\{sensor}\\{method}\\{city}\\{dataType}\\{NA_type}\\{pattern}"
                        path = path_creator(folder,f)
                        plt.vlines(barrier[pattern],0.0001,10,label="Wave barrier",linewidth=2)
                        plt.legend(bbox_to_anchor=(1.01, 0.5),loc='center left',frameon=False)
                        plt.ylabel("Amplitude reduction ratio")
                        plt.xlabel("Distance (m)")
                        plt.xlim((0,20))
                        plt.ylim((0,2))
                        plt.xticks(arange(0,22.5,2.5))
                        #plt.show()
                        plt.savefig(path,dpi=1000,bbox_inches='tight')
                        plt.close()

def fieldvsnumeric():
    layoutDict = {"L2":"L25","L3":"L50"}
    for dataType in ["Accelerations","Velocities"]:
        for F in frequencies:
            for exp in ["A"]:
                for layout in ["L2","L3"]:
                    for location in ["Milas","Menteşe"]:
                        field = readField(dataType,location,layoutDict[layout],exp,F,"NA")
                        numeric = readNumeric(dataType,location,layout,exp,F,"NA")[:len(field)]
                        distance = patterns["S1"][layoutDict[layout]]
                        plt.plot(distance,field,label="Field")
                        plt.plot(distance,numeric,label="Numeric")
                        plt.semilogy()
                        plt.ylabel(f"Normalized {dataType}")
                        plt.xlabel("Distance (m)")
                        plt.title(f"{F} Hz")
                        plt.grid(True)
                        plt.legend()
                        path = f"D:\\Projeler\\Tubitak\\Datas\\Numeric\\{dataType}\\{location}\\{exp}-{layout}"
                        plt.savefig(path_creator(path,F))
                        plt.close()

arg_list = [
    (1,),
    (2,),
    (3,),
    ("avg",)
]

func = AmplitudeReductionRatio
processDict = {}
import multiprocessing
if __name__ == '__main__':
    for i,arg in enumerate(arg_list):
        processDict[i] = multiprocessing.Process(target=func,args=arg)

    for p in processDict:
        processDict[p].start()
    
    for p in processDict:
        processDict[p].join()
