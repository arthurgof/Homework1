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
