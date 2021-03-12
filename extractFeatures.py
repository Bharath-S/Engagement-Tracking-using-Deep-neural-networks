import os
# TODO take pathToOpenFaceBinary as sys argument
pathToOpenFaceBinary = "../../OpenFace/build/bin"

outputTrainVideos = "../Output_CaD_Train_Videos/"
outPutFeatureExtraction = "../Output_Feature_Extraction/"

for video in os.listdir(outputTrainVideos):
        file_name = os.path.join(outputTrainVideos, video)
        os.system(pathToOpenFaceBinary + "/FeatureExtraction -f %s -out_dir %s -2Dfp -3Dfp -pdmparams -pose "
                                         "-aus -gaze"%(file_name, outPutFeatureExtraction))
        os.system("rm %s*.txt"%(outPutFeatureExtraction))