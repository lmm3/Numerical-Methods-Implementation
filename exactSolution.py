from sympy import * 
import matplotlib.pyplot as plt
y, t, x = symbols('y t x')


es = []
tn = []
str_expr = '25*exp(-1*t/25) - 25*exp(-3*t/25)'
expr = sympify(str_expr)
print(expr)
h = 0.1
t0 = 0.0 
f = open('i1.txt', 'w')
for i in range(0, 1000):
    sol = expr.subs(t, t0)
    es.append(sol)
    f.write(str(sol)+'\n')
    t0 += h
    tn.append(t0)
f.close()
plt.title('Solução Exata i1')
plt.plot(tn, es)
plt.ylabel('i1(A)')
plt.xlabel('t(min)')
plt.savefig('Images/SolucaoExataI1')
plt.close()
