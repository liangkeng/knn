import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file = open('data_knn.txt','r')
data_standing = file.readline()
data_walking = file.readline()
data_cycling = file.readline()
#print(data_cycling)
#print(type(json.loads(data_cycling)))

direction_x_standing = []
direction_y_standing = []
direction_z_standing = []
for item in json.loads(data_standing):
    direction_x_standing.append(item['direction_x'])
    direction_y_standing.append(item['direction_y'])
    direction_z_standing.append(item['direction_z'])

direction_standing = np.vstack((direction_x_standing,direction_y_standing,direction_z_standing)).T
#print(direction_standing)

direction_x_walking = []
direction_y_walking = []
direction_z_walking = []
for item in json.loads(data_walking):
    direction_x_walking.append(item['direction_x'])
    direction_y_walking.append(item['direction_y'])
    direction_z_walking.append(item['direction_z'])

direction_walking = np.vstack((direction_x_walking,direction_y_walking,direction_z_walking)).T
# print(direction_walking)

direction_x_cycling = []
direction_y_cycling = []
direction_z_cycling = []
for item in json.loads(data_cycling):
    direction_x_cycling.append(item['direction_x'])
    direction_y_cycling.append(item['direction_y'])
    direction_z_cycling.append(item['direction_z'])

direction_cycling = np.vstack((direction_x_cycling,direction_y_cycling,direction_z_cycling)).T
# print(direction_cycling)

#三种数据合并，并加标签
direction = np.append(direction_standing,direction_walking,axis=0)
direction = np.append(direction,direction_cycling,axis=0)
#print(direction)
target = np.zeros((1,600))
target = np.append(target,np.ones((1,600)))
target = np.append(target,np.full((1,600), 2))

 #将站立、步行、骑车三种状态数据各选600组。并划分为训练数据和测试数据。1800*0.7=1260个训练
direction_train, direction_test, target_train,target_test = train_test_split(direction, target, test_size=0.3)
knn = KNeighborsClassifier()
#进行训练
knn.fit(direction_train,target_train)
#使用训练好的knn进行数据预测
knn_predict = knn.predict(direction_test)
knn_score =knn.score(direction_train,target_train)
# print(knn_predict)
# print(target_test)
print(knn_predict == target_test)
print('knn_score:',knn_score)
# print(1)
