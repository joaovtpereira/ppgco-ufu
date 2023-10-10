# iniciando com python e teste de ambiente

# x = 2 +3
# a = 1
# print(x)
# print(a)

import numpy as np
A = np.matrix([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Teste de acessos de elementos na matrix do numpy

print("Matrix")
print(A)

print("Acessos")
print(A[0,1])
print(A[0,3])
print(A[1,1])

print("Segunda linha e todos os elementos das colunas")
print(A[1,:])
print("Todas as linhas contendo somente a 2 coluna")
print(A[:, 1])

print("Print de acesso de dados da matrix usando o slice")
print(A[0:2, 1:3])
print("Fazendo uma manipulacao mais louca ainda, no caso vou pular uma coluna")
print(A[0:2, 0:4:2])