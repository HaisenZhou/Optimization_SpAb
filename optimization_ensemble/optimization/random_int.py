import random
import math
import pandas as pd
import numpy as np

class random_int:
    def __init__(
                self,
                num_offspring: int = 30
                ):
        self.num_offspring = num_offspring
        
    
    def my_comb(self, n, m):
        return math.factorial(n)//(math.factorial(n-m)*math.factorial(m))

    # Convert a number to the solution of the baffle method
    def num2Comb(self, num, n, m, l=[]):
        if m == 0:
            return l
        if n <= m:
            l.extend([m-i for i in range(m)])
            return l
        if num > self.my_comb(n-1, m):
            num -= self.my_comb(n-1, m)
            l.append(n)
            n -= 1
            m -= 1
        else:
            n -= 1
        return self.num2Comb(num, n, m, l)

    # The content of each component is determined by the position of the separator 
    def comb2List(self, l, n, m):
        temp = [n+1] + l + [0]
        return [temp[i]-temp[i+1]-1 for i in range(len(temp)-1)]

    # Each solution of the baffle method corresponds to a number 
    def genRandom(self, trueM, trueN):
        n = trueN + trueM - 1
        m = trueM - 1
        total = self.my_comb(n, m) 
        # add constraints, X5<40, and, x1+x2+x3+x4+x7>x6+x8
        while True:
            num = random.randint(1, total)
            l = self.num2Comb(num, n, m, [])
            ans = self.comb2List(l, n, m)
          
            if ans[4] < 40 and sum(ans[:4]) + ans[6] > ans[5] + ans[7]:  
                return ans
        
        # num = random.randint(1, total)
        # l = self.num2Comb(num, n, m, [])
        # ans = self.comb2List(l, n, m)
        # return ans
    
    def propose_new_candidates(self):
        # output a list of 100 solutions
        l = []
        for i in range(self.num_offspring):
            ans = self.genRandom(8, 100)
            l.append(ans)
            #print(ans, sum(ans))
        l = np.array(l)
        l = l / l.sum(axis=1, keepdims=True)
        return l

