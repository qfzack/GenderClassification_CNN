import os

# path = r'./gender/female/'
# f = os.listdir(path)
# n = 0
# for i in f:
#     oldname = path+f[n]
#     newname = path+'female'+str(n+1)+'.jpg'
#     os.rename(oldname, newname)
#     print(oldname,'===>',newname)
#     n += 1
#
# path = r'./gender/male/'
# f = os.listdir(path)
# n = 0
# for i in f:
#     oldname = path+f[n]
#     newname = path+'male'+str(n+1)+'.jpg'
#     os.rename(oldname, newname)
#     print(oldname,'===>',newname)
#     n += 1

path = r'./gender/female_validation/'
f = os.listdir(path)
n = 0
for i in f:
    oldname = path+f[n]
    newname = path+'female'+str(n+1)+'.jpg'
    os.rename(oldname, newname)
    print(oldname,'===>',newname)
    n += 1

path = r'./gender/male_validation/'
f = os.listdir(path)
n = 0
for i in f:
    oldname = path+f[n]
    newname = path+'male'+str(n+1)+'.jpg'
    os.rename(oldname, newname)
    print(oldname,'===>',newname)
    n += 1