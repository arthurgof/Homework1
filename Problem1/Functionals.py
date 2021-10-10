cube = lambda x:pow(x,3) # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    lam = list()
    for i in range(n):
        if i < 2:
            lam.append(i)
        else:
            lam.append(lam[i-1]+lam[i-2])
    return lam
