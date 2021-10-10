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
