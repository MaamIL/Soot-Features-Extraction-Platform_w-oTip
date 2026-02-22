# Soot-Profiles-Extraction-Plat
A general platform for creating datasets and running various deep learning models on it for research of extracting soot profiles out of a flame image. Built for Dr. Victor Chernov, Department of Mechanical Engineering, Braude College.
As I'm working for a researcher, changes are done often. The generelization idea is to switch between models and input/output properties easily and fast.

configs before run:
In main.py - 
confogure params for dataset creation and/or model:
#Params for dataset creation    
        
        self.root_dir = where is the data taken from. Note- this should be a folder holding folders of data.
        self.modelpath = for testing of inference - location of model you want to run on (.pth)
        self.MODE = Set to "Train" or "SingleTest" or "Inference" or "TrainSaveAllData" MODE as needed (train- train the model, test- load and test the model on a single sample (input-output), inference- load model and run inference on a single sample (input only))
        
there are normalization params for the dataset with tips and without tips. comment/uncomment accordingly.
if using a different dataset for training- comment these and uncomment the code under:
line 61: #For calculating the above params from the dataset:
Note- this takes some time, depending ob the dataset.

self.model_name = "CNNencdec" - this is the model architecgture you want to use. it can vary through files under Mymodels.

after configuration (make sure files actually exists)- run main.py.




More- TBD

