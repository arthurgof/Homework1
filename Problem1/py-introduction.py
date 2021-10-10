if __name__ == '__main__':
    print("Hello, World!")

	
#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if(n%2==1):
        print('Weird')
    elif(n>=2 and n<=5):
        print('Not Weird')
    elif(n>=6 and n<=20):
        print('Weird')            
    else:
        print('Not Weird')

		
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

	
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(round(a/b))
    print(a/b)

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i)        

def is_leap(year):
    leap = False
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                leap = True
        else:
            leap = True
    return leap

year = int(input())

if __name__ == '__main__':
    n = int(input())
    out = ''
    for i in range(n):
        out+=str(i+1)
    print(out)    
