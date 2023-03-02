import os
import pandas as pd

# FIRST WE ARE GOING TO EXTRACT ALL THE INFORMATION FROM THE .TSV FILES
train_data_path="C:/Users/enriq/Desktop/i-care-international-cardiac-arrest-research-consortium-database-1.0/training/"
output_data="C:/Users/enriq/Desktop/i-care-international-cardiac-arrest-research-consortium-database-1.0/"
process_data = pd.DataFrame()
for filename in os.listdir(train_data_path):
    data=pd.read_csv(os.path.join(train_data_path, filename, filename+".tsv"),sep='\t')
    data_notime=data.drop(columns=["Hour", "Time"])
    data_nonNan=data_notime.dropna()
    process_data=process_data.append(data_nonNan)


# NOW WE ARE GOING TO EXTRACT ALL THE INFORMATION FROM THE .TXT FILES
recordings_names=process_data['Record'].tolist()
Age_list=[]
Sex_list=[]
ROSC_list=[]
OHCA_list=[]
VFib_list=[]
TTM_list=[]
CPC_list=[]
Outcome_list=[]

for filename_rec in recordings_names:
    filename=filename_rec.split("_",2)
    with open(os.path.join(train_data_path,"ICARE_"+filename[1], "ICARE_"+filename[1]+".txt")) as file:
        for line in file:
            s = line.split()
            for i,j in enumerate(s):
                if j == "Age:":
                    Age_list.append(s[i+1])
                elif j == "Sex:":
                    Sex_list.append(s[i+1])
                elif j == "ROSC:":
                    ROSC_list.append(s[i+1])
                elif j == "OHCA:":
                    OHCA_list.append(s[i+1])
                elif j == "VFib:":
                    VFib_list.append(s[i+1])
                elif j == "TTM:":
                    TTM_list.append(s[i+1])
                elif j == "CPC:":
                    CPC_list.append(s[i+1])
                elif j == "Outcome:":
                    Outcome_list.append(s[i+1])

process_data['Age'] = Age_list
process_data['Sex'] = Sex_list
process_data['ROSC'] = ROSC_list
process_data['OHCA'] = OHCA_list
process_data['VFib'] = VFib_list
process_data['TTM'] = TTM_list
process_data['CPC'] = CPC_list
process_data['Outcome'] = Outcome_list


# SAVE IN A .CSV FILE (all data)
process_data.to_csv(os.path.join(output_data,r'DatasetInformationParsed.csv'),index=False)

# FILTERED DATA (Qs>0.7)
#filtered_data

# SAVE IN A .CSV FILE (filtered data)
#filtered_data.to_csv(os.path.join(output_data,r'DatasetInformationFiltered_07.csv'),index=False)