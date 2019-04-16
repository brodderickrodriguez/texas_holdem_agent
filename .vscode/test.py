import math
m = int(input())
found = False
def findPairs( n): 
      
    # find cube root of n 
    cubeRoot = int(math.pow(n, 1.0 / 3.0)); 
  
    # create a array of  
    # size of size 'cubeRoot' 
    cube = [0] * (cubeRoot + 1); 
  
    # for index i, cube[i]  
    # will contain i^3 
    for i in range(1, cubeRoot + 1): 
        cube[i] = i * i * i; 
  
    # Find all pairs in above sorted 
    # array cube[] whose sum  
    # is equal to n 
    l = 1; 
    r = cubeRoot; 

    res = 0
  
    while (l < r): 
        if (cube[l] + cube[r] < n): 
            l += 1; 
        elif(cube[l] + cube[r] > n): 
            r -= 1; 
        else:
            res += 1 
            l += 1; 
            r -= 1; 
            if res == 2:
                return True

    return False
for i in reversed(range(m)):
    found = findPairs(i)
    if found:
        print(i)
        break
else:
    print('none')