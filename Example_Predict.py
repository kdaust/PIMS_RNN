import torch
import torch.nn as nn
import numpy as np

##need to source the GRU_Model
BATCHSIZE = 32
trained_model = torch.load("OldMods/GRU_Mod_C5")
data_input = torch.load("TestData_Input.pt")
grount_truth = torch.load("TestData_GroundTruth.pt")
h = trained_model.init_hidden(BATCHSIZE)
pred = mod(data_input.to("cuda:0").float(), h)
