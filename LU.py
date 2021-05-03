import numpy as np

"""
    @author Arash Alaei <arashalaei22@gmail.com>.
    @since 5/3/2021.
    @description decompose matrix to L AND U.
"""
class LU:

    def __init__(self, matrix):
        self.__original_matrix = matrix
        self.__L = 0
        self.__U = 0
        self.__lu_calculator()
    
    def __lu_calculator(self):
        n = self.__original_matrix.shape[0]
        U = self.__original_matrix.copy()
        L = np.eye(n, dtype=np.double)
        for i in range(n):
            factor = U[i+1:, i] / U[i, i]
            L[i+1:, i] = factor
            U[i+1:] -= factor[:, np.newaxis] * U[i]
        self.__L = L
        self.__U = U
    
    def __forward_substitution(self, b):
        n = self.__L.shape[0]
        y = np.zeros_like(b, dtype=np.double)
        y[0] = b[0] /self.__L[0, 0]
        for i in range(1, n):
            y[i] = (b[i] - np.dot(self.__L[i,:i], y[:i])) / self.__L[i,i]
        return y

    def __back_substitution(self, y):
        n = self.__U.shape[0]
        x = np.zeros_like(y, dtype=np.double)
        x[-1] = y[-1] / self.__U[-1, -1]
        for i in range(n-2, -1, -1):
            x[i] = (y[i] - np.dot(self.__U[i,i:], x[i:])) / self.__U[i,i]
        return x

    def solver(self, b):
        y = self.__forward_substitution(b)
        return self.__back_substitution(y)



# Driver code
if __name__ == '__main__':
    n, m = map(int,input().split(' '))
    A = np.zeros((n, n))
    B = np.zeros((m, n))
    
    for i in range(n):
        l = list(map(float,input().strip().split()))[:n]
        A[i] = l
    
    for i in range(m):
        l = list(map(float,input().strip().split()))[:n]
        B[i] = l

    lu = LU(A)

    ans = np.zeros((m, n))

    for i in range(m):
        ans[i] = lu.solver(B[i])


    for i in range(m):
        for j in range(n):
            print(ans[i][j], end=' ')
        print()





