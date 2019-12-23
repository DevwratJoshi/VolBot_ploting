import os

url = './test'
os.chdir(url)
for f in os.listdir():
    a = f.split('_')
    if(int(a[3]) > 3):
        os.remove(f)
