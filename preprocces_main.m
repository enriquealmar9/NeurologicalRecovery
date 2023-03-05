%LOAD THE FILES 
train_path="C:/Users/enriq/Desktop/i-care-international-cardiac-arrest-research-consortium-database-1.0/training/"
DatasetInformationFiltered=load('DatasetInformationFiltered.mat');
DatasetInformationFiltered=DatasetInformationFiltered.DatasetInformationFiltered;
output_path="C:/Users/enriq/Desktop/i-care-international-cardiac-arrest-research-consortium-database-1.0/training_preprocces"
for c = 1:height(DatasetInformationFiltered)
    full_name=string(DatasetInformationFiltered{c,1});
    general_name=split(full_name,'_');
    general_name=strcat(general_name(1,1),"_",general_name(2,1));
    recording=load(strcat(train_path,general_name,"/",full_name,".mat"));
    recording=recording.val;
    out=preprocessing_signal(recording);
    v_min=min(min(out));
    out=out+abs(v_min);
    v_max=max(max(out));
    out=(out/v_max)*(1600);
    out=out-800;
    output_path=pwd;
    fileID = fopen(strcat(full_name,"_pre.raw.tif"),'w');
    fwrite(fileID, out,'uint16');
    fclose(fileID);
    
end