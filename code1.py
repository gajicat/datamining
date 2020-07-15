import numpy as np # 加载numpy工具库并给它取个别名为np，后面就可以通过np来调用numpy工具库里面的函数了。numpy是python的一个科学计算工具库，
                    # 除了前面文章中提到的它可以用来进行向量化之外，它还有很多常用的功能。非常非常常用的一个工具库！
import matplotlib.pyplot as plt # 这个库是用来画图的

import h5py # 这个库是用来加载训练数据集的。我们数据集的保存格式是HDF。Hierarchical Data Format(HDF)是一种针对大量数据进行组织和存储的
            #  文件格式,大数据行业和人工智能行业都用它来保存数据。
import skimage.transform as tf # 这里我们用它来缩放图片


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r") # 加载训练数据
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # 从训练数据中提取出图片的特征数据
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # 从训练数据中提取出图片的标签数据

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r") # 加载测试数据
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:]) # 加载标签类别数据，这里的类别只有两种，1代表有猫，0代表无猫
        
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # 把数组的维度从(209,)变成(1, 209)，这样好方便后面进行计算
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0])) # 从(50,)变成(1, 50)
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 30
plt.imshow(train_set_x_orig[index])
print ("标签为" + str(train_set_y[:, index]) + ", 这是一个'" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' 图片.")


# 我们要清楚变量的维度，否则后面会出很多问题。下面我把他们的维度打印出来。
print ("train_set_x_orig shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_orig shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1] # 由于我们的图片是正方形的，所以长宽相等

print ("训练样本数: m_train = " + str(m_train))
print ("测试样本数: m_test = " + str(m_test))
print ("每张图片的宽/高: num_px = " + str(num_px))

# 为了方便后面进行矩阵运算，我们需要将样本数据进行扁平化和转置
# 处理后的数组各维度的含义是（图片数据，样本数）

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w, b
    
def propagate(w,b,X,Y):
    m = X.shape[1]
    A=sigmoid(np.dot(w.T, X)+b)
    cost=-np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)) / m
    dZ = A - Y
    dw = np.dot(X, dZ.T)/m
    db = np.sum(dZ)/m
    grads = {"dw":dw,
             "db":db}
    return grads, cost
    
def optimize(w,b,X,Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("优化%i次后成本是: %f" %(i, cost))
    params = {"w":w,
              "b":b}
    return params, costs

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0,i] >= 0.5:
            Y_prediction[0,i] = 1
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):    
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    print("训练准确率：{}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("测试准确率：{}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs":costs,
         "Y_prediction_test":Y_prediction_test,
         "Y_prediction_train":Y_prediction_train,
         "w":w,
         "b":b,
         "learning_rate":learning_rate,
         "num_iterations":num_iterations
         }
         
    return d
	
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

index = 8
plt.imshow(test_set_x[:,index].reshape((num_px,num_px, 3)))
print("这张图标签"+str(test_set_y[0,index])+" 预测结果"+str(int(d["Y_prediction_test"][0,index])))

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title("learning rate =" + str(d["learning_rate"]))
plt.show()