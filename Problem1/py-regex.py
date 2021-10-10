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
