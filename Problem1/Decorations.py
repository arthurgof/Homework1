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

