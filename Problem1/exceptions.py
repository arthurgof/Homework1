N = int(input())
for i in range(N):
    try:
        tr = input().split()
        a = int(tr[0])
        b = int(tr[1])
        print(a//b)
    except Exception as e:
        print("Error Code:",e)
