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
	
		

