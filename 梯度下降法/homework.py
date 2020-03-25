import numpy as np
from numpy.linalg import norm

# 函數 f 對變數 k 的偏微分: df / dk
def df(f, p, k, step=0.01):
    p1 = p.copy()
    p1[k] = p[k]+step
    return (f(p1) - f(p)) / step

# 函數 f 在點 p 上的梯度
def grad(f, p, step=0.01):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k, step)
    return gp

# 使用梯度下降法尋找函數最低點
def gradientDescendent(f, p0, step=0.01):
    p = p0.copy()
    i = 0
    fp0 = f(p)
    while (True):
        i += 1
        fp = f(p)
        gp = grad(f, p) # 計算梯度 gp
        glen = norm(gp) # norm = 梯度的長度 (步伐大小)
        print('{:d}:p={:s} f(p)={:.3f} gp={:s} glen={:.5f}'.format(i, str(p), fp, str(gp), glen))
        if glen < 0.00001 or fp0 < fp:  # 如果步伐已經很小了，或者 f(p) 變大了，那麼就停止吧！
            break
        gstep = np.multiply(gp, -1*step) # gstep = 逆梯度方向的一小步
        p +=  gstep # 向 gstep 方向走一小步
        fp0 = fp
    return p # 傳回最低點！

import gd2 as gd
import numpy as np

A = np.array([[3.0,2,1],[1,4,-3],[2,-3,2]])
B = np.array([[1.0,2,-1]]).transpose()



def f(p): 
    X = p
    Y = A.dot(X)
 
    return np.linalg.norm(Y-B, 1)


p = np.array([0.0,0,0])
gd.gradientDescendent(f, p, step=0.01)
