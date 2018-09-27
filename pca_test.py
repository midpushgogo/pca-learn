import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pca(data,dimension):
    mean=np.mean(data,axis=0) # 对列求均值
    data=data-mean # 中心化
    cov=np.cov(data,rowvar=0) #协方差矩阵
    vals,vects=np.linalg.eig(cov) # 提取特征值和特征向量
    valsort=np.argsort(vals)[:-dimension-1:-1] # 提取n个最大特征值的序号
    eigvects=vects[:,valsort] # 对应的特征向量
    lowdata=np.dot(data,eigvects) # 基代换
    return lowdata

iris = datasets.load_iris() #读取数据
data = iris['data']
features = iris.data.T
ax = plt.subplot(111, projection='3d')
ax.scatter(features[0],features[1],features[2],alpha=0.9,s=50*features[3],c=iris.target,cmap='viridis')
ax.view_init(elev=60, azim=80)
plt.show() # 用点大小作为第四维显示

# 降维处理 2维
res = pca(data,2)
features3 = res.T
plt.scatter(features3[0],features3[1],alpha=0.9,c=iris.target,cmap='viridis')
plt.show()