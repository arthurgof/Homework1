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
	