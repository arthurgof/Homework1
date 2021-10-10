t, N = map(int,input().split())
pt = list()
for i in range(N):
    pt.append(map(float,input().split()))

v = zip(*pt)
for i in v:
    print(sum(i)/N)

	
N, n = map(int, input().split())
li = [list(map(int, input().split())) for i in range(N)]
par = int(input())
li.sort(key=lambda x : x[par])
for i in li:
    print(*i)


tr = [x for x in input()]
tr = sorted(tr, key = lambda x : ord(x))
tr = sorted(tr, key = lambda x : x in '02468')
tr = sorted(tr, key = lambda x : -ord(x) >> 5)
print(*tr,sep='')
def wrapper(f):
    def fun(li):
        for i in range(len(li)):
            x = li[i] 
            p1 = [n for n in x][-10:-5]
            p2 = [n for n in x][-5:]
            st = '+91 '
            st += ''.join(p1)+' '
            st += ''.join(p2)
            li[i] = st
        f ([n for n in li])
    return fun



def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key = lambda x:int(x[2])))
    return inner

N = int(input())
for i in range(N):
    try:
        tr = input().split()
        a = int(tr[0])
        b = int(tr[1])
        print(a//b)
    except Exception as e:
        print("Error Code:",e)
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
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    ml = []
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                if not i+j+k == n:
                    ml.append([i,j,k])
    print(ml)

	
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr = list(arr)
    # sort algo not efficient but easy to code O(n) = n^2 could use Merge Sort if we want a better algo
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i == j:
                continue            
            if arr[i] == arr[j]:
                arr[i] = -200
            if arr[i] > arr[j]:
                tr = arr[i]
                arr[i] = arr[j]
                arr[j] = tr
    print(arr[1])

	
if __name__ == '__main__':
    dct = {}
    w1 = 100000
    w2 = 100000
    for _ in range(int(input())):
        name = input()
        score = float(input())
        if dct.get(score)!=None:
            dct[score].append(name)
        else:
            dct[score] = [name]
        if score < w1:
            w2 = w1
            w1 = score
        elif w2 > score and not score==w1:
            w2 = score
    an = list(dct[w2])
    an.sort()
    for i in an:
        print(i)

		
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    arr = student_marks[query_name]
    sc = 0
    for i in arr:
        sc+=i
    sc/=len(arr)
    print("{0:.2f}".format(sc))

	
if __name__ == '__main__':
    N = int(input())
    print(N)

	
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    integer_list = list(integer_list)
    for i in range(len(integer_list)):
        integer_list[i] = int(integer_list[i])
    T = tuple(integer_list)
    print(hash(T))
from collections import Counter as cn

tr = input()
seq = list(map(int,input().split()))
coun = cn(seq)
N = int(input())
tot = 0
for i in range(N):
    tai, cost = map(int,input().split())
    if coun[tai]:
        tot+= cost
        coun[tai] -= 1
        
print(tot)


from collections import defaultdict as dd

d = dd(list)
n,m = map(int, input().split())

for i in range(n):
    d[input()].append(i+1)
    
for i in range(m):
    v = input()
    if v in d:
        print(*d[v])
    else:
        print(-1)

		
from collections import namedtuple

N = int(input())
l = input().split()
studs = namedtuple('student',l)
tot = 0
for i in range(N):
    var = input().split()
    stud = studs(var[0],var[1],var[2],var[3])
    tot += int(stud.MARKS)
    
print('{:.2f}'.format(tot/N))
        

from collections import OrderedDict

N = int(input())
dic = OrderedDict()
for i in range(N):
    item, space, price = input().rpartition(' ')
    if dic.get(item):
        dic[item] += int(price) 
    else:
        dic[item] = int(price)
for i in dic:
    print(i+" "+str(dic[i]))

	
from collections import OrderedDict

N = int(input())
dic = OrderedDict()
for i in range(N):
    st = input()
    if dic.get(st):
        dic[st] += 1
    else:
        dic[st] = 1
v = list(dic.values())
print(len(v))
print(*v)


from collections import deque
N = int(input())
d = deque()
for i in range(N):
    st = input().split()
    if len(st)==1:
        if st[0] =="popleft":
            d.popleft()
        else:
            d.pop()
    else:
        eval(f'd.{st[0]}({st[1]})')
print(*d)


from collections import OrderedDict

from collections import Counter

st = input()
N = [char for char in st]
cn = Counter(N)
l1 = sorted(cn.items(), key=lambda c: (-c[1], c[0]))
l1 = l1[:3]
for i in l1:
    print(i[0]+" "+str(i[1]))

	
# TURING MACHINE ?
N = int(input())
for i in range(N):
    n = int(input())
    ltr = list(map(int,input().split()))
    i1 = 0
    i2 = n - 1
    lastb = -222
    for y in range(n):
        if lastb == -222:
            if ltr[i1] < ltr[i2]:
                lastb = ltr[i2]
                i2 -= 1
            else :
                lastb = ltr[i1]
                i1 +=1
            continue
        if lastb < ltr[i1] & lastb < ltr[i2]:
            print('No')
            break
        else:
            if ltr[i1] < ltr[i2] & ltr[i2] <= lastb:
                lastb = ltr[i2]
                i2 -= 1
            elif ltr[i1] <= lastb:
                lastb = ltr[i1]
                i1 +=1
            else:
                print('No')
                break
            if i1 > i2:
                print('Yes')
                break
import calendar
day,mounth,year = map(int,input().split())
alday = list(calendar.day_name)
print(alday[calendar.weekday(year, day, mounth)].upper())


from datetime import datetime
N = int(input())
formate = '%a %d %b %Y %H:%M:%S %z'
for i in range(N):
    tr1 = input()
    tr2 = input()
    diff = datetime.strptime(tr1,formate) - datetime.strptime(tr2,formate)
    print(abs(int(diff.total_seconds())))
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
N = int(input())
l = list()
for i in range(N):
    try:
        n = input()
        a = float(n)
        float(n.split('.')[1])
        l.append(True)
    except Exception as e:
        l.append(False)
for i in l:
    print(i)

	

regex_pattern = r"[,\.]"	# Do not delete 'r'.



import re

sl = input()
pa = r'([a-zA-Z0-9])\1+'
m = re.search(pa, sl)
if m:
    print(m.group(1))
else:
    print(-1)

	

import re

vo = "aeiouAEIOU"
co = "ZRTYPQSDFGHJKLMWXCVBNzrtypqsdfghjklmwxcvbn"
m = re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (co, vo, co), input())
if len(m)==0:
    print(-1)
for i in m:
    print(i)

	
	
import re

st = input()
k = input()
m = re.search(k,st)
if m:
    index = 0
    while index + len(k) < len(st):
        try:
            m = re.search(k,st[index:])
            s = index+m.start()
            f = index+m.end() - 1
            print(f'({s}, {f})')
            index += 1 + m.start()
        except:
            break
else:
    print('(-1, -1)')

	
	
import re

N = int(input())
for i in range(N):
    st = input()
    while ' && ' in st or ' || 'in st:
        k1 = r' (&&) '
        k2 = r' (\|\|) '
        st = re.sub(k1, " and ", st)
        st = re.sub(k2, " or ", st)
    print(st)

	
	
upperbound = '(M{0,3})(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})'
regex_pattern = r''+upperbound+'$'	# Do not delete 'r'.



import re

N = int(input())
for i in range(N):
    st = input()
    pa = r'[789]\d{9}$'
    m = re.match(pa,st)
    if m:
        print('YES')
    else:
        print('NO')
    

import email.utils
import re

N= int(input())
pa = r'[A-Za-z](\w|-|\.|_)+@+[a-zA-Z]+\.+[a-zA-Z]{1,3}$'
for i in range(N):
    st = input()
    em = email.utils.parseaddr(st)[1]
    m = re.match(pa,em)
    if m:
        print(st) 

		
		
import re

N = int(input())
allow = False
pa =r'#[a-fA-F0-9]{3,6}[;,)]'
for i in range(N):
    st = input()
    if allow:
        m = re.findall(pa,st)
        if len(m) > 0:
            for y in m:
                print(y[:-1])
    elif "{" in st:
        allow = True
    elif "}" in st:
        allow = False

		
		
from html.parser import HTMLParser
# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for i in attrs:
            print('->',i[0],'>',i[1])
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print ('Empty :',tag)
        for i in attrs:
            print('->',i[0],'>',i[1])

N = int(input())
parser = MyHTMLParser()
for i in range(N):
    parser.feed(input())

	
	
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        ltr = len(data.split('\n'))
        if ltr == 1:
            print('>>> Single-line Comment')
        else:
            print('>>> Multi-line Comment')
        if data.strip():
            print(data)
    
    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

N = int(input())
pa = MyHTMLParser()
li = list()
for i in range(N):
    st = input()
    li.append(st)
pa.feed('\n'.join(li))



from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for i in attrs:
            print('-> '+i[0]+' > '+i[1])
n = int(input())
li = [input() for i in range(n)]
tr = '\n'.join(li)
hp = MyHTMLParser()
hp.feed(tr)



import re

pa1 = r'.*[A-Z].*[A-Z].*'
pa2 = r'.*(\d).*(\d).*(\d).*'
pa3 = r'[\w]{10}'
pa4 = r'.*(.).*\1.*'
N = int(input())
for i in range(N):
    st = input()
    m1 = re.search(pa1,st)
    m2 = re.search(pa2,st)
    m3 = re.search(pa3,st)
    m4 = re.search(pa4,st)
    if m1 and m2 and m3 and not m4 :
        print('Valid')
    else:
        print('Invalid')

		
		
import re

pa1 = r'([0-9]{16}|((([0-9]{4})+\-){3}[0-9]{4}))$'#correct numbers
pa2 = r'([\d])\1\1\1'
pa3 = r'[456]{1}'
N = int(input())
for i in range(N):
    st = input()
    m1 = re.match(pa1,st)
    m3 = re.match(pa3,st)
    st = st.replace('-','')
    m2 = re.search(pa2,st)
    if m1 and m3 and not m2 :
        print('Valid')
    else:
        print('Invalid')

		
		
regex_integer_in_range = r"[1-9][\d]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=.\1)"	# Do not delete 'r'.


#!/bin/python3

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = list()
pa = r'\b[^\w]+\b'

for _ in range(n):
    matrix_item = [char for char in input()]
    matrix.append(matrix_item)
st = ""    
for i in range(m):
    for y in range(n):
        st += matrix[y][i]
print(re.sub(pa, ' ',st))
def average(array):
    return float(sum(sorted(set(array)))/(len(sorted(set(array)))))

	
tr = input()
obtain = input().split()
plus = set(input().split())
minu = set(input().split())
pos = sum(i in plus for i in obtain)
neg = sum(i in minu for i in obtain)
print(pos - neg)


st = set()
re = set()
for i in range(2):
    tr = input()
    v = set(map(int,input().split()))
    st.update(v)
    if i==0:
        re.update(v)
    else:
        re  = re.intersection(v)
st = sorted(st.difference(re))
for i in st:
    print(i)

	
# Enter your code here. Read input from STDIN. Print output to STDOUT
pos = int(input())
dist = set()
for i in range(pos):
    dist.add(input())
print(len(dist))


pos = int(input())
dis = set(map(int,input().split()))
size = int(input())
for i in range(size):
    ls = input().split()
    if ls[0]=='remove':
        dis.remove(int(ls[1]))
    elif ls[0]=='discard':
        dis.discard(int(ls[1]))  
    else:
        dis.pop()
print(sum(dis))


v = set()
tr = input()
v = v.union(set(input().split()))
tr = input()
v = v.union(set(input().split()))
print(len(v))


tr = input()
v = set(input().split())
tr = input()
v = v.intersection(set(input().split()))
print(len(v))


tr = input()
v = set(input().split())
tr = input()
v = v.difference(set(input().split()))
print(len(v))


tr = input()
v = set(input().split())
tr = input()
v = v.symmetric_difference(set(input().split()))
print(len(v))


tr = input()
v = set(input().split())
N = int(input())
for i in range(N):
    s = input().split()
    se = set(input().split())
    eval(f'v.{s[0]}(se)')
s = 0
for i in v:
    s += int(i)
print(s)


n = int(input())
seq = list(map(int,input().split()))
se = set(seq)
dif = (sum(se)*n - sum(seq))
print(int(dif/(n-1)))


N = int(input())
for i in range(N):
    tr = input()
    A = set(input().split())
    tr = input()
    B = set(input().split())
    print(len(B) == len(B.union(A)))

	
B = set(input().split())
N = int(input())
v = True
for i in range(N):
    ns = set(input().split())
    v = v & (len(B) == len(B.union(ns)))
print(v)
	def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)
	


def split_and_join(line):
    sent = line.split()
    ans = ""
    for i in sent:
        if sent.index(i)==0:
            ans += i
        else:    
            ans += "-"+i
    return ans

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

	
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    print("Hello "+ first+ " " +last+"! You just delved into python.")

def mutate_string(string, position, character):
    tr = [i for i in string]
    tr.insert(position, character)
    tr.pop(position + 1)
    return ''.join(tr)

def count_substring(string, sub_string):
    string = [i for i in string]
    sub_string = [i for i in sub_string]
    count = 0
    for i in range(len(string)):
        for y in range(len(sub_string)):
            if i+y >= len(string):
                break
            elif not string[i+y] == sub_string[y]:
                break
            elif y == len(sub_string) - 1:
                count+=1
                
    return count

if __name__ == '__main__':
    s = input()
    testcases = ['isalnum', 'isalpha', 'isdigit', 'islower', 'isupper']
    for test in testcases:
        print(any(eval("i." + test + "()") for i in s))

		
n = int(input())
si = 'H'
for i in range(n):
    print(si.rjust(n-(i))+(si*i*2))
    
for i in range(n+1):
    print((si*n).center(n*2)+(si*n).center(n*6))
    
for i in range((n+1)//2):
    print((si*5*n).center(n*6))
    
for i in range(n+1):
    print((si*n).center(n*2)+(si*n).center(n*6))
    
for i in range(n):
    print(' '+si.rjust(n*4+i)+(si*(n-(i+1))*2))
	


def wrap(string, max_width):
    ls = [char for char in string]
    st = ""
    id1 = 0
    for i in ls:
        id1 += 1
        st += i
        if id1 == max_width:
            st += "\n"
            id1 = 0
    return st

n, m = map(int, input().split())
pa = list()
for i in range(n//2):
    tr = ('.|.'*(2*i + 1)).center(m, '-')
    pa.append(tr)
for i in pa:
    print(i)
print('WELCOME'.center(m, '-'))
for i in range(len(pa)-1,-1,-1):
    print(pa[i])

	
def print_formatted(number):
    out = []
    for i in range(number):
        tr = i +1
        n = str(tr)
        oc = str(oct(tr)[2:])
        hexa = str(hex(tr)[2:]).upper()
        bi = str(bin(tr)[2:])
        out.append([n, oc, hexa, bi])
        
    width = len(out[-1][3])
    for i in out:
        print(*(char.rjust(width) for char in i))

		
def print_rangoli(size):
    tra = 'abcdefghijklmnopqrstuvwxyz'
    li = list()
    for i in range(size):
        tr = "-".join(tra[i:size])
        li.append((tr[::-1]+tr[1:]).center(4*size-3, "-"))
    for i in range(len(li)-1,0,-1):
        print(li[i])
    for i in range(len(li)):
        print(li[i])



# Complete the solve function below.
def solve(s):
    tr = [i for i in s]
    tr[0] = tr[0].upper()
    for i in range(len(tr)):
        if tr[i] == " ":
            tr[i+1] = tr[i+1].upper()
    return ''.join(tr)


def minion_game(string):
    vowels =['A','E','I','O','U']
    vowel=0
    cons=0
    for i in range(len(string)):
        if string[i] in vowels:
            vowel+= len(string)-i
        else:
            cons+=len(string)-i
    if cons>vowel:
        print("Stuart "+ str(cons))
    elif vowel>cons:
        print("Kevin "+ str(vowel))
    else:
        print("Draw")
        

# for earlier exercise
def count_substring(string, sub_string):
    string = [i for i in string]
    sub_string = [i for i in sub_string]
    count = 0
    for i in range(len(string)):
        for y in range(len(sub_string)):
            if i+y >= len(string):
                break
            elif not string[i+y] == sub_string[y]:
                break
            elif y == len(sub_string) - 1:
                count+=1
                
    return count

	
def merge_the_tools(string, k):
    for i in range(0,len(string),k):
        sub = ""
        for y in range(k):
            if not string[y+i] in sub:
                sub += string[y+i]
        print(sub)
	
		

def get_attr_number(node):
    #transform xml to byte
    tree = etree.tostring(node)
    #transfom byte to string
    tr = str(tree)
    #each of the element of score have an equal
    tot = tr.count('=')
    return tot




maxdepth = 0
def depth(elem, level):
    global maxdepth
    if len(elem) > 0: #if it is not a leaf
        level += 1
        maxdepth = max(level + 1, maxdepth)
    for i in elem:
        # go trough all child
        depth(i, level)




#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    c = 0
    m = 0
    for i in candles:
        if m < i:
            m = i
            c = 0
        if m == i:
            c += 1
    return c
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2, inti = True):
    if inti:
        if x1 > x2:
            return kangaroo(x2, v2, x1, v1, False)
        elif v2 > v1 or v1 == v2:
            return 'NO'
    if (x2 - x1)% (v2-v1)==0:
        return 'YES'
    return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    nt = 5
    nv = 0
    for i in range(n):
        nv += nt//2
        nt = (nt//2)*3
    return nv
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    n = str(k*sum([int(x) for x in n]))
    if len(n) == 1:
        return n
    else:
        return superDigit(n,1)
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def cprint(arr):
    st = ""
    for i in range(len(arr)):
        if i == 0:
            st += str(arr[i])
        else:
            st += " " + str(arr[i])
    print(st)

def insertionSort1(n, arr):
    basev = arr[-1]
    for i in range(len(arr) - 2, -1, -1):
        if arr[i] > basev:
            arr[i + 1] = arr[i]
            cprint(arr)
        else:
            arr[i + 1] = basev
            cprint(arr)
            break
    if arr[0] > basev:
        arr[0] = basev
        cprint(arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

	
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def cprint(arr):
    st = ""
    for i in range(len(arr)):
        if i == 0:
            st += str(arr[i])
        else:
            st += " " + str(arr[i])
    print(st)

def insertionSort2(n, arr):
    for i in range(len(arr)):
        if i==0:
            continue
        index = i - 1
        va = arr[i]
        while index >= 0 and arr[index] > va:
            arr[index + 1] = arr[index]
            index -= 1
        arr[index+1] = va
        cprint(arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

