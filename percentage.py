values = []
methods = ['Euler', 'Euler Inverso', 'Euler Aprimorado', 'Runge-Kutta', 'Adams-Bashforth', 'Adams-Multon', 'Fórmula Inversa']
orders = ['1', '1', '2', '4', '5', '6', '6']
f = open('comparison.txt', 'r')
for i in range(0,7):
    values.append(float(f.readline()))
trueValue = float(f.readline())
f.close
f = open('percentages.txt', 'w')
for i in range(0, 7):
    difference = abs(trueValue - values[i])
    percentage = difference/abs(trueValue)*100.0
    f.write(methods[i] + ' & ' + str(values[i]) + ' & ' + str(percentage) + '\\% & ' + orders[i] + ' \\\\' + '\n')
    f.write('\\hline\n')

f.close
    