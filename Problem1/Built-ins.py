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
