#!/usr/bin/python
# _*_ encoding=utf-8

import xlrd as xlrd
#open the file
from My_Hero.Description_State import DescriptionState
import matplotlib.pyplot as plt
import numpy as np

def printMSN(MSN_LIST):
    file_object = open('C:\\Users\\lenovo\\Desktop\\data_class_AZ.txt', 'w')
    for list in MSN_LIST:
         for j in list:
             file_object.write(str(j)+"\n")
            #(j+"\n")
        # print("\n")
         file_object.write("\n")


def MSN_Description(AZ):
    AZ_MSN = []
    AZ_TMP = []
    for i in AZ:
        if i.Year < 2009:
            AZ_TMP.append(i)
        else:
            AZ_TMP.append(i)
            AZ_MSN.append(AZ_TMP)
            AZ_TMP = []

    return AZ_MSN

data=xlrd.open_workbook("C:\\Users\\lenovo\\Desktop\\ProblemCData.xlsx")

table = data.sheets()[0]
data_list = []
print(table.nrows)
data_list.extend(table.row_values(0))
#打印出第一行的全部数据


#print(description_State.speak())
#print(description_State)
#wb = load_workbook()
#print(wb.sheetnames)
#description sheet
sheet_name=["seseds","msncodes"]
#sheet = wb.get_sheet_by_name("seseds")
#description the row
row_key=["A","B","C","D","E"]

#print/save variable
#the four State list save infomation node


#Attributes for the node
MSN=""
StateCode=""
Year=0
Data=0.0

AZ = []
CA = []
NM = []
TX = []
for i in range(1,table.nrows):
    MSN = table.row_values(i)[0]
    StateCode = table.row_values(i)[1]
    Year = table.row_values(i)[2]
    Data = table.row_values(i)[3]
    description_state = DescriptionState(MSN, StateCode, Year, Data)
    if StateCode == 'AZ':
        AZ.append(description_state)
    if StateCode == 'CA':
        CA.append(description_state)
    if StateCode == 'NM':
        NM.append(description_state)
    if StateCode == 'TX':
        TX.append(description_state)
#sort of year
AZ.sort(key=DescriptionState.Year)
CA.sort(key=DescriptionState.Year)
NM.sort(key=DescriptionState.Year)
TX.sort(key=DescriptionState.Year)
#classification if MSN use ASCII
AZ.sort(key=DescriptionState.MSN)
CA.sort(key=DescriptionState.MSN)
NM.sort(key=DescriptionState.MSN)
TX.sort(key=DescriptionState.MSN)

#split the MSN
AZ_MSN = MSN_Description(AZ)
CA_MSN = MSN_Description(CA)
NM_MSN = MSN_Description(NM)
TX_MSN = MSN_Description(TX)

printMSN(TX_MSN)
#print(len(AZ_MSN))
#print(len(CA_MSN))
#print(len(NM_MSN))
#print(len(TX_MSN))


strs="WYTCB"

AZ_year = []
AZ_data=[]

for j in AZ_MSN:
    if j[0].MSN == strs:
        for i in j:
            AZ_year.append(i.Year)
            AZ_data.append(i.Data)

CA_year = []
CA_data=[]

for j in CA_MSN:
    if j[0].MSN == strs:
        for i in j:
            CA_year.append(i.Year)
            CA_data.append(i.Data)
NM_year = []
NM_data=[]

for j in NM_MSN:
    if j[0].MSN == strs:
        for i in j:
            NM_year.append(i.Year)
            NM_data.append(i.Data)
TX_year = []
TX_data=[]

for j in TX_MSN:
    if j[0].MSN == strs:
        for i in j:
            TX_year.append(i.Year)
            TX_data.append(i.Data)

#for i in year:
    #print(i)
#for i in data:
    #print(i)
plt.plot(TX_year,TX_data,'r')
plt.plot(CA_year,CA_data,'y')
plt.plot(AZ_year,AZ_data,'g')
plt.plot(NM_year,NM_data,'b')
plt.show()



#for i in AZ:
    #if i.Year < 2009:
       # AZ_TMP.append(i)
    #else:
       # AZ_TMP.append(i)
       # AZ_MSN.append(AZ_TMP)
        #AZ_TMP = []

#for list in AZ_MSN:
    #for j in list:
    #    print(j)
   # print("\n")
#index = 0;
#for i in AZ:
#    print(i)

#print(index)


