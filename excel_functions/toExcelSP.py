from numpy import loadtxt,arange,array,mean,max,abs,where
from pandas import DataFrame,ExcelWriter,read_excel
import os

frequencies = arange(10,160,10)
depths = [4,6,8,12,18]
distances = {
    "L1":[1,1.5,2,3.5,6,8.5,11,13.5,16,18.5,21,25,29],
    "L2":[1,2,3.5,6,8.5,11,13.5,16,18.5,21,25,29],
    "L3":[1,2.5,6,8.5,11,13.5,16,18.5,21,25,29],
    "L4":[1,4.75,8.75,11,13.5,16,18.5,21,25,29],
    "L5":[1,6,8.5,11,13.5,16,18.5,25,29],
}

def read_file(model_name,dataType):
    path = f"D:\\Projeler\\Tubitak\\Output\\SP_Output\\{model_name}"
    datas = loadtxt(os.path.join(path,f"{model_name}_{dataType}.txt"),skiprows=1,unpack=True)[1:]
    return datas

def readNA(dataType,city,pattern,exp,depth,frequency):
    if exp=="A":
        path = f"D:\\Projeler\\Tubitak\\Output\\Excel\\{dataType}\\3 - Normalized Values\\{city}\\{pattern}.xlsx"
        excel = read_excel(path,sheet_name="A")
    else:
        path = f"D:\\Projeler\\Tubitak\\Output\\Excel\\SP\\{dataType}\\3 - Normalized Values\\{city}\\{pattern}.xlsx"
        excel = read_excel(path,sheet_name=f"{depth}m")
    F_col = excel.iloc[:,0]
    index = where(F_col==frequency)[0][0]
    return excel.iloc[index,1:].values

def path_creator(path):
    path = os.path.join("Datas", "NewGraphs", "Filtered", path)
    if not os.path.exists(path):
        os.makedirs(path)

def maxFunc(dataType,city,depth,pattern,frequency):
    model_name = f"{city}_SP_{depth}m_{pattern}_{frequency}Hz_2D"
    data = read_file(model_name,dataType)
    return sorted(max(abs(data),axis=1),reverse=True)

def readAttenuation(dataType,city,pattern,F):
    model_name = f"{city}_A_{pattern}_{F}Hz_2D"
    path = f"D:\\Projeler\\Tubitak\\Output\\2D\\{model_name}"
    datas = loadtxt(os.path.join(path,f"{model_name}_{dataType}.txt"),skiprows=1,unpack=True)[1:]
    return sorted(max(abs(datas),axis=1),reverse=True)

def DFFunc(dataType,city,depth,pattern):
    Max,NA,AR = {},{},{}
    for F in frequencies:        
        barrierMax = maxFunc(dataType,city,depth,pattern,F)
        Max[F] = array(barrierMax)
        normalised = Max[F]/Max[F][0]
        NA[F] = normalised
        max_A = readAttenuation(dataType,city,pattern,F)
        NA_A = max_A/max_A[0]
        
        ar = normalised/NA_A
        AR[F]=ar

    DF_max = DataFrame.from_dict(Max, orient="index", columns=[f"C{i+1}" for i in range(len(Max[10]))])
    DF_NA = DataFrame.from_dict(NA, orient="index", columns=[f"C{i+1}" for i in range(len(NA[10]))])
    DF_AR = DataFrame.from_dict(AR, orient="index", columns=[f"C{i+1}" for i in range(len(AR[10]))])
    return DF_max,DF_NA,DF_AR

def toExcel(city,dataType):
    for pattern in ["L1","L2","L3","L4","L5"]:
        Max_path =  f"D:\\Projeler\\Tubitak\\Output\\Excel\\SP\\{dataType}\\2 - Max Values\\{city}"
        NA_path =  f"D:\\Projeler\\Tubitak\\Output\\Excel\\SP\\{dataType}\\3 - Normalized Values\\{city}"
        AR_path =  f"D:\\Projeler\\Tubitak\\Output\\Excel\\SP\\{dataType}\\4 - AR(Conventional Method)\\{city}"
        path_creator(Max_path)
        path_creator(NA_path)
        path_creator(AR_path)
        writer_max = ExcelWriter(os.path.join(Max_path,f"{pattern}.xlsx"), engine='xlsxwriter')
        writer_NA = ExcelWriter(os.path.join(NA_path,f"{pattern}.xlsx"), engine='xlsxwriter')
        writer_AR = ExcelWriter(os.path.join(AR_path,f"{pattern}.xlsx"), engine='xlsxwriter')
        for d in depths:
            print(dataType,city,f"{d}-{pattern}",dataType)
            Max,NA,AR = DFFunc(dataType,city,d,pattern)
            Max.to_excel(writer_max,sheet_name=f"{d}m")
            NA.to_excel(writer_NA,sheet_name=f"{d}m")
            AR.to_excel(writer_AR,sheet_name=f"{d}m")
            
            
        writer_max.save()
        writer_NA.save()
        writer_AR.save()

#Direct Excel
def direct(secondSensor,dataType,location,depth,pattern,frequency):
    NA_barrier = readNA(dataType,location,pattern,"SP",depth,frequency)
    NA_attenuation = readNA(dataType,location,pattern,"A",0,frequency)
    AR_list = []

    for firstSensor in [1,2,3]:
        AR = (NA_barrier[secondSensor-1]*NA_attenuation[firstSensor-1]) /(NA_barrier[firstSensor-1]*NA_attenuation[secondSensor-1])
        AR_list.append(AR)
    AR_list.append(mean(AR_list))
    return AR_list

def directDF(dataType,city,depth,pattern):
    AR = {}
    startFrom = 5 if pattern=="L5" else 4
    seconds = arange(startFrom,len(distances[pattern])+1)
    for f in frequencies:
        columns = []
        temp = []
        for s in seconds:
            columns.extend([f"S{s}(1)",f"S{s}(2)",f"S{s}(3)",f"S{s}(avg)"])
            temp.extend(direct(s,dataType,city,depth,pattern,f))
        AR[f]=temp
    DF_AR = DataFrame.from_dict(AR, orient="index", columns=columns)
    return DF_AR

def directExcel(city,dataType):
    path =  f"D:\Projeler\Tubitak\Output\Excel\SP\{dataType}\\5 - AR(Direct Method)\\{city}"
    path_creator(path)
    for pattern in ["L1","L2","L3","L4","L5"]:
        fileName = f"{pattern}.xlsx"
        writer_AR = ExcelWriter(os.path.join(path,fileName), engine='xlsxwriter')
        for depth in depths:
            print(dataType,city,f"{depth}-{pattern}",dataType)
            AR = directDF(dataType,city,depth,pattern)
            AR.to_excel(writer_AR,sheet_name=f"{depth}m")
            
        writer_AR.save()

#Direct Average Excel
def readAR(dataType,location,pattern,depth,region):
    path = f"D:\\Projeler\\Tubitak\\Output\\Excel\\SP\\{dataType}\\5 - AR(Direct Method)\\{location}\\{pattern}.xlsx"
    data = []
    indexes = {"near" : [4,8],"far":[12,16]}
    ind = indexes[region]
    for F in frequencies:
        excel = read_excel(path,sheet_name=f"{depth}m")
        F_col = excel.iloc[:,0]
        index = where(F_col==F)[0][0]
        data.append(mean(excel.iloc[index,ind].values))
    
    return data

def avgFunc(city,dataType,depth,region):
    AR = {}
    for L in ["L1","L2","L3","L4","L5"]:
        AR[L] = readAR(dataType,city,L,depth,region)
    
    DF = DataFrame.from_dict(AR, orient="index", columns=frequencies)
    return DF

def avgExcel(city,dataType):
    path =  f"D:\\Projeler\\Tubitak\\Output\\Excel\\SP\\{dataType}\\6 - AR Direct (Avg)"
    print(city,dataType)
    path_creator(path)
    for region in ["far","near"]:
        writer = ExcelWriter(os.path.join(path,f"{city}-{region}.xlsx"), engine='xlsxwriter')
        for depth in depths:
            DF = avgFunc(city,dataType,depth,region)
            DF.to_excel(writer,sheet_name=f"{depth}m")
        writer.save()


def run(func):
    arg_list = [
        ("Milas","Acc"),
        ("Milas","Vel"),
        ("Mentese","Vel"),
        ("Mentese","Acc"),
        ("Senaryo1","Vel"),
        ("Senaryo1","Acc"),
        ("Senaryo2","Vel"),
        ("Senaryo2","Acc"),
        ("Senaryo3","Vel"),
        ("Senaryo3","Acc"),
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

run(avgExcel)