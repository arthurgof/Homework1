

def arrays(arr):
    num = numpy.array(arr, float)
    num = num[::-1]
    return num


	
import numpy as np
lst = list(map(int,input().split()))
arr = np.array(lst)
print(np.reshape(arr,(3,3)))



import numpy as np
n, m = map(int, input().split())
array = np.array([list(map(int,input().split())) for i in range(n)])
print (array.transpose())
print (array.flatten())



import numpy as np
n, m = map(int, input().split())
array = np.array([list(map(int,input().split())) for i in range(n)])
print (array.transpose())
print (array.flatten())



import numpy as np

n, m, p = map(int, input().split())

arr1 = np.array([list(map(int, input().split())) for _ in range(n)])
arr2 = np.array([list(map(int, input().split())) for _ in range(m)])
print(np.concatenate((arr1, arr2), axis = 0))


import numpy as np
dim = list(map(int, input().split()))
print(np.zeros(dim, dtype = int))
print(np.ones(dim, dtype = int))


import numpy as np

dim1, dim2  = map(int, input().split())
print(str(np.eye(dim1, dim2, k = 0)).replace('1',' 1').replace('0',' 0'))



import numpy as np
N, n = map(int, input().split())
arr1 = np.array([list(map(int, input().split())) for _ in range(N)])
arr2 = np.array([list(map(int, input().split())) for _ in range(N)])
print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 * arr2)
print(arr1 // arr2)
print(arr1 % arr2)
print(arr1 ** arr2)


import numpy as np

ls = input().split()
n = -3
print('[',str(np.floor(np.array(ls, float))).replace('.','. ')[1:n]+'.]')
print('[',str(np.ceil(np.array(ls, float))).replace('.','. ')[1:n]+'.]')
print('[',str(np.rint(np.array(ls, float))).replace('.','. ')[1:n]+'.]')



import numpy as np

N, M = map(int, input().split())
arr = np.array([list(map(int, input().split())) for _ in range(M)])
print(np.prod(np.sum(arr, axis = 0)))


import numpy as np

N, M = map(int, input().split())
arr = np.array([list(map(int, input().split())) for _ in range(M)])
print(np.min(np.max(arr, axis = 0))) 


import numpy as np

N, M = map(int, input().split())
arr = np.array([list(map(int,input().split())) for _ in range(M)])
print(np.mean(arr, axis = 1))
print(np.var(arr, axis = 0))
np.set_printoptions(legacy='1.14')
print(round(np.std(arr, axis = None),11))


import numpy as np

N = int(input())
arr1 = np.array([list(map(int, input().split())) for _ in range(N)])
arr2 = np.array([list(map(int, input().split())) for _ in range(N)])
print(np.dot(arr1, arr2))


import numpy as np

arr1 = np.array([list(map(int, input().split()))])
arr2 = np.array([list(map(int, input().split()))])
print(np.inner(arr1,arr2)[0][0])
print(np.outer(arr1,arr2))


import numpy as np

arr = list(map(float, input().split()))
x = float(input())
print(np.polyval(arr, x))


import numpy as np

N = int(input())
arr1 = np.array([list(map(float, input().split())) for _ in range(N)])
np.set_printoptions(legacy='1.13')
print(np.linalg.det(arr1))
