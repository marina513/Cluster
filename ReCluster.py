import numpy as np

def ResetDictIndex(Dict):
    DictNew = {}
    for i,key in enumerate(Dict):
        DictNew[i] = Dict[key]
    return DictNew



# Given path & its corresponding embedding
def Recluster(paths, embeddings, threshold):
    # Initial Clusters
    ClusterMeans = {} ; ClusterEmbeds = {} ; ClusterPaths = {}
    for i in range(len(paths)):
        ClusterEmbeds[i] = [embeddings[i]]
        ClusterPaths[i] = [paths[i]]
        ClusterMeans[i] = np.mean(ClusterEmbeds[i], axis=0) 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CurrentIDMatch_value = 1 ; iter_ = 0
    while(CurrentIDMatch_value > threshold): # Fake love break (both match each other but @ Low value) case 1
        print("iter : " , iter_)
        for CurrentID in range(len(ClusterPaths)):
            
            if(CurrentID >= len(ClusterPaths)): # to overcome out of index due to deleted clusters case 0
                break

            means = np.array(list(ClusterMeans.values())) 
            means = means.reshape(means.shape[0], 512 ) 

            # Current embed most match eachother
            CurrentIDMatch_ID  = np.argsort(ClusterMeans[CurrentID].dot(means.T))[-2]
            CurrentIDMatch_Match_ID  = np.argsort(ClusterMeans[CurrentIDMatch_ID].dot(means.T))[-2]

            #print(CurrentID, CurrentIDMatch_ID, CurrentIDMatch_Match_ID)
            if(CurrentID==CurrentIDMatch_Match_ID): # love match 
                #print("Love Match @ ", CurrentIDMatch_value)
                # both clusters match @ what value
                CurrentIDMatch_value  = np.sort(ClusterMeans[CurrentID].dot(means.T))[-2]
                
                # Merge
                ClusterPaths[CurrentID]   = ClusterPaths[CurrentID]  + ClusterPaths[CurrentIDMatch_ID]
                ClusterEmbeds[CurrentID]  = ClusterEmbeds[CurrentID] + ClusterEmbeds[CurrentIDMatch_ID]
                ClusterMeans[CurrentID] = np.mean(ClusterEmbeds[CurrentID], axis=0)

                # Delete merged
                del ClusterPaths[CurrentIDMatch_ID]   ;  ClusterPaths = ResetDictIndex(ClusterPaths)
                del ClusterEmbeds[CurrentIDMatch_ID]  ;  ClusterEmbeds = ResetDictIndex(ClusterEmbeds)
                del ClusterMeans[CurrentIDMatch_ID]   ;  ClusterMeans = ResetDictIndex(ClusterMeans)
        iter_ += 1 
    return ClusterPaths, ClusterMeans, ClusterEmbeds