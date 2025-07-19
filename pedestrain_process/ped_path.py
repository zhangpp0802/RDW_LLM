import pickle as pkl
import pandas as pd
idx = 6
inFile = "/depot/bera89/data/zhan5058/TUTR/pedestrain_process/006.pkl"
outFile = "/depot/bera89/data/zhan5058/TUTR/pedestrain_process/006_processed.pkl"
with open(inFile, "rb") as f:
    object = pkl.load(f)

df = pd.DataFrame(object)
# print(df.keys()) #['scene_id', 'time', 'goal', 'p1', 'p2']
scene_id = df["scene_id"]
time = df["time"]
goal = df["goal"]
p1 = df["p1"]
p2 = df["p2"]
print(p1[0]['head']['pos'][:,:2].shape)

# # print(len(p1))
# # print(len(p2))
# cols = ['scene_id', 'time', 'goal', 'p1', 'p2']
# for i in range(len(p1)):
#     # print(p1[i].keys())
#     del p1[i]['left hand']
#     del p1[i]['right hand']
#     del p1[i]['waist']
#     del p1[i]['left foot']
#     del p1[i]['right foot']

#     del p2[i]['left hand']
#     del p2[i]['right hand']
#     del p2[i]['waist']
#     del p2[i]['left foot']
#     del p2[i]['right foot']

# # print(p1[0].keys())
# # print(type(scene_id))
# data = [scene_id,time,goal,p1,p2]

# new_df = pd.DataFrame(data, columns=cols)
# p1 = new_df['p1']
# print(new_df.keys())
# # print(p1[0]['head']['pos'][:,:2]) #(641, 2)
# # new_df['p1'] = p1[0]['head']['pos'][:,:2]
# # print()
# new_df.to_pickle(outFile)

# # # verify outputfile working correctly.
# with open(outFile, "rb") as f:
#     object = pkl.load(f)
# df = pd.DataFrame(object)
# p1 = df["p1"]
# # exit()
# print(df.keys())
# print(p1[0].keys())
# print(p1[0]['head']['pos'][:,:2].shape)