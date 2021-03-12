import os
import ffmpeg

inputTrainVideos = "../../Train/"
outputTrainVideos = "../Output_CaD_Train_Videos/"

for video in os.listdir(inputTrainVideos):
    pathToInputVideo = os.path.join(inputTrainVideos, video)
    pathToOutputVideo = os.path.join(outputTrainVideos, video)

    (ffmpeg
    .input(pathToInputVideo)
    .trim(start=30)
    .setpts('PTS-STARTPTS')
    .trim(duration=270)
    .filter('fps', fps=10, round='up')
    .output(pathToOutputVideo)
    .run()
     )
