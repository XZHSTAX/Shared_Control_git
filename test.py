import numpy as np
def how_sim(a1,a2):
    X = np.array([[2,1,1,1,0,0],
                  [1,2,1,0,0,0],
                  [1,1,2,0,0,1],
                  [1,0,0,2,1,1],
                  [0,0,0,1,2,1],
                  [0,0,1,1,1,2]])
    return X[a1,a2] 

print(how_sim(1,[0,1,2,3,4,5])) 
a = np.array([0,0,1,1])
print(np.argmax(a))
print(np.random.rand(3))