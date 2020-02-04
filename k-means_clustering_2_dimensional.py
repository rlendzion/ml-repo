# k-means clustering for 2 dimensional datasets
# a step-by-step explanation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr = np.empty((0,2), dtype=float)
arr = np.append(arr, np.array([[18,113800],[22,215000],[35,400000],[12,110000],[15,136700],
                               [33,580100],[15,203300],[35,498100],[45,521300],[20,232300],
                               [51,607100],[48,520100],[24,200200],[19,199900],[45,704800],
                               [63,50034],[66,66066],[70,20020],[72,79665],[77,50470]]), axis=0)

# print the size of dataset
print(np.shape(arr))

# normalize data for better performance
j = 0
while j < 2:
    minimum = min(arr[:, j])
    maximum = max(arr[:, j])
    for i in range(0,len(arr)):
        arr[i,j] = (arr[i,j]-minimum)/(maximum-minimum)
    j += 1
print(arr)

# specify number of centroids
n = 3
carr = np.empty((0,2), dtype=float)

# initialize centroids
np.random.seed(999)
carr = np.append(carr, np.random.rand(n,2), axis=0)
print(carr)

e = 0
while e < 20:
    # display data and centroids on scatter plot
    plt.scatter(arr[:,0],arr[:,1])
    plt.scatter(carr[:,0],carr[:,1], c='red')
    if e == 0:
        plt.title('Centroids after initialization')
    else:
        plt.title('Centroids after '+str(e)+' epochs')
    plt.show()

    # calculate distance from every observation to each centroid
    dis = np.empty((0,n))
    l = 0
    while l < n:
        for i in range(0,len(arr)):

            # calculate Euclidean distance from each centroid
            dis = np.append (dis, np.array([[i+1,'centroid_'+str(l+1),np.sqrt(pow(arr[i,0]-carr[l,0],2)+pow(arr[i,1]-carr[l,1],2))]]), axis=0)

        l += 1

    # convert results to DataFrame
    df = pd.DataFrame(dis, columns=['inx','centr','calc'])
    df['inx'] = df['inx'].astype(int)
    df['calc'] = df['calc'].astype(float)

    # transpose data
    pivot = pd.pivot_table(df, index='inx', columns = ['centr'], aggfunc=np.sum)

    # drop a multi-level column index and obsolete columns
    pivot.columns = pivot.columns.droplevel(0)
    pivot = pivot.reset_index()
    pivot = pivot.drop(['inx'], axis=1)

    # get the nearest cluster name per observation and append to the original data
    pivot['cluster'] = pivot.idxmin(axis=1)
    pivot = pivot['cluster']
    a = np.array(pivot)
    a = np.reshape(a, (20,1))
    concat = pd.DataFrame(np.concatenate((arr,a),axis=1), columns=['x','y','cluster'])
    concat['x']=concat['x'].astype(float)
    concat['y']=concat['y'].astype(float)
    concat['cluster']=concat['cluster'].str.replace('[^0-9]+', '')

    # get new centroids
    concat_x = concat.groupby('cluster').agg({'x':['mean']})
    concat_y = concat.groupby('cluster').agg({'y':['mean']})
    new_centroids = np.concatenate((concat_x,concat_y),axis=1)

    # cache old centroids
    cache = carr

    # overwrite old centroids
    carr = new_centroids

    # update loop parameter
    e += 1

    # stop loop if centroids no longer change
    if cache[0,0]==carr[0,0]:
        break

    # print the data with assigned clusters
    print(concat)

    # notification
    print('Process finished:\r\n\tIt took ',e,' epochs to get the final centroids')
