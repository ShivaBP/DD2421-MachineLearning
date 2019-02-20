import numpy as np
import random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt 

N = None
Targets = None 
Data = None 
alpha = None
classA = None 
classB = None 
sIndex = 0
kernelSelector = 2   # 0: linnear  1:polynomial  2: radial basis 

def genData():
    
    global N
    global Targets
    global Data
    global alpha
    global classA
    global classB
    global sIndex

    # generate data 
    np.random.seed(100)
    classA = np.concatenate(
        (np.random.randn(10, 2) * 0.2 + [1.5 , 0.5], 
        np.random.randn(10 , 2 )* 0.2 + [- 1.5 , 0.5] )
    )
    classB = np.random.randn(20, 2) * 0.2 + [0.0 , -0.5 ]

    Inputs = np.concatenate((classA, classB))
    Outputs = np.concatenate(
        (np.ones(classA.shape[0]) , 
        -np.ones(classB.shape[0])
        )
    )  
    
    N = Inputs.shape[0]
    permute = list (range(N))
    random.shuffle(permute)
    Data = Inputs[permute ,  :]
    Targets = Outputs[permute]

    # impose constraints , a dictionary 
    XC = {'type':'eq' , 'fun': zerofun}

    # start is a vector with initial guess of alpha values 
    start = np.zeros(N)

    # lower and upper bopund tuple for each alpha values 
    C = 1 #controls the margin 
    B = [(0, C) for b in range(N)]

    # create alpha 
    ret = minimize(objective, start , bounds=B , constraints=XC)
    alpha = ret['x']

    #check for linnear seperability 
    if ( ret['success']  ):
        print("success")
    else:
        print("fail")

    # thresholding
    threshold = math.pow(10 , (-5))
    for i in range(N):
        if (alpha[i] < threshold):
            alpha[i] = 0 

    # choosiong a support vector 
    sIndex = 0
    for i in range(N):
        if (alpha[i] != 0 ):
            sIndex = i
            break

# return the scalar of two datapoints 
def kernel(point1 , point2):

# if linnear kernel 
    if (kernelSelector == 0 ):
        point1T = np.transpose(point1)
        result = np.dot(point1T , point2)

# if polynomial kernel 
    elif (kernelSelector ==1 ):
        polynomDegree = 2 
        point1T = np.transpose(point1)
        dotProduct = np.dot(point1T , point2)
        result  = math.pow((dotProduct +1) , polynomDegree )

# if radial basis kernel 
    elif(kernelSelector == 2):
        sigma = 1
        difference  = np.subtract(point1 , point2)
        vectorNorm = np.linalg.norm(difference)
        part1  = math.pow( vectorNorm , 2)
        part2 = 2 * math.pow(sigma , 2)
        exponent  =  -part1 / part2
        result  = math.pow( math.e  , exponent )

    return result 

def init():
    
   #initialize the P matrix
    P = np.zeros((N, N))
    # Fill in the P matrix 
    for i in range(N):
        for j in range(N):
            kernelValue = kernel(Data[i] , Data[j])
            P[i][j] = Targets[i] * Targets[j] * kernelValue
    return P
    
def objective(alpha):

    PMatrix = init()
    sum1 = 0 
    for i in range(N):
        for j in range(N):
           sum1 = sum1 +  alpha[i] * alpha[j] * PMatrix[i][j]
    sum1 = sum1/2 
    sum2 = 0
    for i in range(N):
        sum2 = sum2 + alpha[i]
    result = sum1 - sum2
    return result

def zerofun(alpha):

    result  = 0
    for i in range (N):
        result = result + alpha[i]*Targets[i]
    return result 

def bias():
    
    sTarget = Targets[sIndex]
    sum = 0
    for i in range(N):
        kernelValue = kernel(Data[sIndex] , Data[i])
        sum = sum + alpha[i]* Targets[i]* kernelValue 
    biasValue = sum - sTarget
    return biasValue

    
def inidicator(x , y):
    
    sv = [x , y]
    s = Data[sIndex]
    sum= 0
    for i in range(N):
        kernelValue = kernel(sv , Data[i])
        sum = sum + alpha[i]*Targets[i]*kernelValue
    indicator = sum - bias()
    return indicator

def plot():

    # the data 
    plt.plot([p[0] for p in classA] ,[p[1] for p in classA]  , 'b.')
    plt.plot([p[0] for p in classB] ,[p[1] for p in classB]  , 'r.')
    plt.axis('equal')  #force same scale on both axes

    # the descision boundary
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array( [ [  inidicator(x, y ) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid , grid  , ( -1 , 0, 1) , colors = ('red' , 'black' , 'blue') , linewidths = (1,3,1) )

    plt.savefig('radialKernel_sigma1.pdf')  #save a copy in file
    plt.show()

def run():

    genData()
    plot()