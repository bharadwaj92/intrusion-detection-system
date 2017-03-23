import numpy as np
a = ['touch'
,'touch'
,'touch'
,'touch'
,'touch'
,'touch'
,'touch'
,'touch'
,'cat'
,'ls'
,'ksh'
,'ls'
,'ls'
,'ksh'
,'ksh'
,'cat'
,'whereis'
,'netstat'
,'netscape'
,'ls']
d = {}
c = 0
for i in a:
    if i in d.keys():
        continue
    else:
        d[i]= c
        c += 1
print(d)
corr_array = np.zeros((len(d), len(d)), int)
for i in range(0, len(a)):
    temp_arr = a[i:i+7]
    visited= []
    print(temp_arr)
    for j in range(1,len(temp_arr)):
        if temp_arr[j] not in visited:
            corr_array[d[temp_arr[0]]][d[temp_arr[j]]] += temp_arr[1:].count(temp_arr[j])
            visited.append(temp_arr[j])
    print("matrix for items in range", i, i+7)
    print(corr_array)
    
print(corr_array)   
    
