import monkdata as m
import dtree as d
import drawtree_qt5 as draw
import random as r
import matplotlib.pyplot as plt

def entropyCalculation():
    print()
    print("Entropy Results", "\n")

    monk1Entropy = d.entropy(m.monk1)
    print("MONK1: ", monk1Entropy, "\n")
  
    monk2Entropy = d.entropy(m.monk2)
    print("MONK2: ", monk2Entropy, "\n")
   
    monk3Entropy = d.entropy(m.monk3) 
    print("MONK3: ", monk3Entropy, "\n")
    print()

def informationGainCalculation():

    print("Information gain results ", "\n")
    for attributeIndex in range(0, 6):
        result = d.averageGain(m.monk1 , m.attributes[attributeIndex])
        print("Monk1|   ", attributeIndex+1 , ": ", result, "    ") 
    print("Best attribute: ", d.bestAttribute(m.monk1 , m.attributes) , "\n")

    for attributeIndex in range(0, 6):
        result = d.averageGain(m.monk2 , m.attributes[attributeIndex])
        print("Monk2|   " ,attributeIndex+1 , ": ", result, "    ") 
    print("Best attribute: ", d.bestAttribute(m.monk2 , m.attributes), "\n")

    for attributeIndex in range(0, 6):
        result = d.averageGain(m.monk3 , m.attributes[attributeIndex])
        print("Monk3|   " , attributeIndex+1 , ": ", result, "    ") 
    print("Best attribute: ", d.bestAttribute(m.monk3 , m.attributes), "\n")      

def buildTree(showTree):

    print("Error computation results ", "\n")
    t1 = d.buildTree(m.monk1 , m.attributes)
    print("MONK1 training: ",  d.check(t1 , m.monk1))
    print("MONK1 test: ",  d.check(t1 , m.monk1test), "\n")
    if (showTree == True):
        draw.drawTree(t1)

    t2 = d.buildTree(m.monk2 , m.attributes)
    print("MONK2 training: ", d.check(t2 , m.monk2))
    print("MONK2 test: ", d.check(t2 , m.monk2test), "\n")
    if (showTree == True):
        draw.drawTree(t2)

    t3 = d.buildTree(m.monk3 , m.attributes)
    print("MONK3 training: ", d.check(t3 , m.monk3))
    print("MONK3 test: ", d.check(t3 , m.monk3test), "\n")
    if (showTree == True):
        draw.drawTree(t3)

def partition(data , fraction):

    datalist= list(data)
    r.shuffle( datalist)
    breakpoint = int( len(data) * fraction )
    return datalist[:breakpoint], datalist[breakpoint:]

def pruning(dataset , fraction ):

    training, validation = partition(dataset , fraction)
    tree = d.buildTree(training , m.attributes)
    pruningchoices = d.allPruned(tree)
    maxIters = 100

    validationScore = 0
    bestChoice = pruningchoices[0]
    iter = 0
    while (iter <= maxIters):
        for prune in pruningchoices:
            if (d.check(prune , validation )  > validationScore):
                validationScore = d.check(prune , validation  )
                bestChoice = prune
        iter = iter + 1    
     
    return bestChoice

def plotPruneAccuracy(dataset , test ):

    print("Pruning accuracy results ", "\n")
    fractions = [0.3, 0.4, 0.5, 0.6 , 0.7 , 0.8]
    fractionResults = list()

    bestFraction = fractions[0]
    bestFractionScore = 0
    for fraction in fractions:     
        prune = pruning(dataset , fraction)
        result = d.check(prune , test )
        fractionResults. append(result )
        plt .plot(fraction, result, 'ro')
        if (d.check(prune , test )  > bestFractionScore):
            bestFractionScore = d.check(prune , test )  
            bestFraction = fraction

    plt.xlabel("Fraction")
    plt.ylabel("Classification Accuracy score on a test set")
    plt.title("Classisfication accuracy of the pruned tree on test data as a function of partitioning fraction")
    plt.show()
    
    print("Best partitioning fraction: ", bestFraction , "\n")
    print("Fraction score: ", bestFractionScore, "\n")
    return fractionResults

def statistics(results):
    print("Statistical computation results" , "\n")
    sum = 0
    for result in results:
        sum = sum + result 
    
    mean = sum / len(results)

    variance= 0
    for result in results:
        localVariance =  (result - mean)**2
        variance = variance + localVariance
    
    print("Mean:    ", mean, "\n")
    print("Variance ", variance , "\n")


def run():
    
   entropyCalculation()
   informationGainCalculation()
   buildTree(False)
   print("Monk1", "\n")
   monk1Results = plotPruneAccuracy(m.monk1 , m.monk1test)
   statistics(monk1Results)

   print()

   print("Monk3", "\n")
   monk3Results = plotPruneAccuracy(m.monk3 , m.monk3test)
   statistics(monk3Results)
   

    
if __name__ == '__main__':
    run()