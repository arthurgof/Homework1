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
