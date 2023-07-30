import os, pickle
import numpy as np

from ReCluster import Recluster
from ClipVec import ImgList2Vector
from SimCLRVec import ImgList2VectorSimCLR
import shutil



def Cluster (Main_dir, use_CLIP ,  use_SimCLR, merge, OutDir):

    ClusterPathsCLIP = None ; ClusterPathsSimCLR = None ; ClusterPathsALL = None
    Main_Imgs_path = Main_dir + "bestShots/"

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    #  Vectorize & Cluster using CLIP
    if(use_CLIP):
        CLIPpath = Main_dir + "/CLIP.pkl"
        
        # get CLIP embedding of all paths
        if(not os.path.exists(CLIPpath)):
            paths = [Main_Imgs_path  + i for  i in os.listdir(Main_Imgs_path)]
            ImgList2Vector(paths , CLIPpath)


        # Read & Cluster
        with open(CLIPpath, 'rb') as fp:
            dfCLIP = pickle.load(fp)
        embeddingsCLIP = np.array( [ np.array(i.cpu()) for i in  list(dfCLIP.values())] )
        pathsCLIP = list(dfCLIP.keys())

        ClusterPathsCLIP, ClusterMeansCLIP, ClusterEmbedsCLIP = Recluster( pathsCLIP , embeddingsCLIP , 0.9)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    #  Vectorize & Cluster using SimCLR
    if(use_SimCLR):
        SimCLRpath = Main_dir + "/SimCLR.pkl"
        
        # get SimCLR embedding of all paths
        if(not os.path.exists(SimCLRpath)):
            ImgList2VectorSimCLR(Main_dir , Main_dir)

        # Read & Cluster
        with open(SimCLRpath, 'rb') as fp:
            dfSimCLR = pickle.load(fp)

        embeddingsSimCLR = np.array(  list(dfSimCLR.values())  )
        pathsSimCLR = list(dfSimCLR.keys())

        ClusterPathsSimCLR, ClusterMeansSimCLR, ClusterEmbedsSimCLR = Recluster( pathsSimCLR , embeddingsSimCLR , 0.9)


    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    # Use both embeddings
    if(merge):
        ClusterPathsALL, ClusterMeansALL, ClusterEmbedsALL =\
                    Recluster( pathsSimCLR , embeddingsSimCLR+embeddingsCLIP , 0.99)
        

    


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    # Save
    Clusters = [ClusterPathsCLIP, ClusterPathsSimCLR , ClusterPathsALL]
    ClustersNames = ["ClIP/", "SimCLR/" , "Merge/"]

    for c in range(len(Clusters)):
        try:
            currDir = OutDir + ClustersNames[c]
            os.mkdir(currDir)
            for sub_c in Clusters[c]:
                
                dir_=  currDir + str(sub_c) + "/"
                os.mkdir(dir_)
                for j in Clusters[c][sub_c]:
                    new = dir_ + "Cluster_" + str(sub_c) + "_name_" + j.split("/")[-1] 
                    shutil.copyfile(j, new)
        
        except:
            pass