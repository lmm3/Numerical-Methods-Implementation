import matplotlib.pyplot as plt
from sympy import *
y, t, x = symbols('y t x');

vt = []
vy = []
fn = []
pc = []

def print_solution(method, h):
    write_file = open('saida.txt', 'a')
    write_file.write('Metodo ' + method + '\n')
    write_file.write('y(' + str(vt[0]) + ') = ' + str(vy[0]) + '\n')
    write_file.write('h = ' + str(h) + '\n')
    for i in range(0, len(vy)):
        write_file.write(str(i) + ' ' + str(vy[i]) + '\n')
    write_file.write('\n')
    write_file.close()

def print_graph(graphName):
    plt.plot(vt, vy)
    plt.ylabel('i1(A)')
    plt.xlabel('t(min)')
    plt.savefig('Images/' + graphName)
    plt.close()

def solveFor(yn, tn, expr):
    return expr.subs([(y,yn), (t,tn)])

def euler_method(y0, t0, h, n, expr):
    vt.append(t0)
    vy.append(y0)
    tn = t0
    yn = y0

    for i in range(1, n + 1):
        yn = yn + h*solveFor(yn, tn, expr)
        tn = tn + h
        vt.append(tn)
        vy.append(yn)

def backward_euler_method(y0, t0, h, n, expr):
    vt.append(t0)
    vy.append(y0)
    tn = t0
    yn = y0
    for i in range(1, n+1):
        yn_euler = yn + h*(solveFor(yn, tn, expr))
        yn = yn + h*(solveFor(yn_euler, tn+h, expr))       
        tn = tn + h
        vt.append(tn)
        vy.append(yn)

def improved_euler_method(y0, t0, h, n, expr):
    vt.append(t0)
    vy.append(y0)

    tn = t0
    yn = y0

    for i in range(1, n+1):
        yn = yn + (h/2)*((solveFor(yn, tn, expr)) + (solveFor(yn + (h * (solveFor(yn, tn, expr))), tn + h, expr)))
        tn = tn + h
        
        vt.append(tn)
        vy.append(yn)

def runge_kutta_method(y0, t0, h, n, expr):
    vt.append(t0)
    vy.append(y0)

    tn = t0
    yn = y0

    for i in range(1, n+1):
        kn1 = solveFor(yn, tn, expr)
        kn2 = solveFor(yn + (h/2)*kn1, tn + (h/2), expr)
        kn3 = solveFor(yn + (h/2)*kn2, tn + (h/2), expr)
        kn4 = solveFor(yn + h*kn3, tn + h, expr)

        yn = yn + (h/6)*(kn1 + 2*kn2 + 2*kn3 + kn4)
        tn = tn + h

        vt.append(tn)
        vy.append(yn)

def adam_bashforth_method(y0, t0, h, n, expr, order):
    yn = y0
    tn = t0

    for i in range(order, n+1):
        fn.clear()
        vt.append(tn)
        for j in range(0, order):
            fn.append(solveFor(vy[len(vy)-1-j], vt[len(vt)-2-j], expr))
        pos = len(fn)-1
        if order == 2:
            yn = yn + h*((3/2)*(fn[pos-1]) - (1/2)*(fn[pos]))
        elif order == 3:
            yn = yn + h*((23/12)*(fn[pos-2]) - (4/3)*(fn[pos-1]) + (5/12)*(fn[pos]))
        elif order == 4:
            yn = yn + h*((55/24)*(fn[pos-3]) - (59/24)*(fn[pos-2]) + (37/24)*(fn[pos-1]) - (3/8)*(fn[pos]))
        elif order == 5:
            yn = yn + h*((1901/720)*(fn[pos-4]) - (1387/360)*(fn[pos-3]) + (109/30)*(fn[pos-2]) - (637/360)*(fn[pos-1]) + (251/720)*(fn[pos]))
        elif order == 6:
            yn = yn + h*((4277/1440)*(fn[pos-5]) - (2641/480)*(fn[pos-4]) + (4991/720)*(fn[pos-3]) - (3649/720)*(fn[pos-2]) + (959/480)*(fn[pos-1]) - (95/288)*(fn[pos]))
        elif order == 7:
            yn = yn + h*((198721/60480)*(fn[pos-6]) - (18637/2520)*(fn[pos-5]) + (235183/20160)*(fn[pos-4]) - (10754/945)*(fn[pos-3]) + (135713/20160)*(fn[pos-2]) - (5603/2520)*(fn[pos-1]) + (19087/60480)*(fn[pos]))
        elif order == 8:
            yn = yn + h*((16083/4480)*(fn[pos-7]) - (1152169/120960)*(fn[pos-6]) + (242653/13440)*(fn[pos-5]) - (296053/13440)*(fn[pos-4]) + (2102243/120960)*(fn[pos-3]) - (115747/13440)*(fn[pos-2]) + (32863/13440)*(fn[pos-1]) - (5257/17280)*(fn[pos]))
        tn = tn + h
        vy.append(yn)
    
    return yn

def adam_multon_method(y0, t0, h, n, expr, order):
    yn = y0
    tn = t0
    
    for i in range(order-1, n+1):
        if order == 2:
            yn1_predicted = yn + h*(solveFor(yn, tn-h, expr))
        else:
            yn1_predicted = adam_bashforth_method(yn, tn, h, order-1, expr, order-1)
        fn1_predicted = solveFor(yn1_predicted, tn, expr)
        pos = len(fn)-1
        if order == 2:
            yn = yn + h*((1/2)*(fn1_predicted) + (1/2)*(solveFor(yn, tn-h, expr)))
            vy.append(yn)
        elif order == 3:
            yn = yn + h*((5/12)*(fn1_predicted) + (2/3)*(fn[pos-1]) - (1/12)*(fn[pos]))
        elif order == 4:
            yn = yn + h*((3/8)*(fn1_predicted) + (19/24)*(fn[pos-2]) - (5/24)*(fn[pos-1]) + (1/24)*(fn[pos]))
        elif order == 5:
            yn = yn + h*((251/720)*(fn1_predicted) + (323/360)*(fn[pos-3]) - (11/30)*(fn[pos-2]) + (53/360)*(fn[pos-1]) - (19/720)*(fn[pos]))
        elif order == 6:
            yn = yn + h*((95/288)*(fn1_predicted) + (1427/1440)*(fn[pos-4]) - (133/240)*(fn[pos-3]) + (241/720)*(fn[pos-2]) - (173/1440)*(fn[pos-1]) + (3/160)*(fn[pos]))
        elif order == 7:
            yn = yn + h*((19087/60480)*(fn1_predicted) + (2713/2520)*(fn[pos-5]) - (15487/20160)*(fn[pos-4]) + (586/945)*(fn[pos-3]) - (5737/20160)*(fn[pos-2]) + (263/2520)*(fn[pos-1]) - (863/60480)*(fn[pos]))
        elif order == 8:
            yn = yn + h*((5257/17280)*(fn1_predicted) + (139849/120960)*(fn[pos-6]) - (4511/4480)*(fn[pos-5]) + (123133/120960)*(fn[pos-4]) - (88574/120960)*(fn[pos-3]) + (1537/4480)*(fn[pos-2]) - (11351/120960)*(fn[pos-1]) + (275/24192)*(fn[pos]))
        vy.pop()
        vy.append(yn)
        tn = tn + h

def backward_formula_method(y0, t0, h, n, expr, order):
    yn = y0
    tn = t0
    if order == 2:
        tn = vt[len(vt)-1]
        backward_euler_method(yn, tn, h, n, expr)
    else:
        for i in range(order-1, n+1):
            yn1_predicted = adam_bashforth_method(yn, tn, h, order-1, expr, order-1)
            fn1_predicted = solveFor(yn1_predicted, tn, expr)
            vy.pop()
            pos = len(vy)-1
            if order == 3:
                yn = (4/3)*vy[pos] -(1/3)*vy[pos-1] + (2/3)*h*fn1_predicted
            elif order == 4:
                yn = (18/11)*vy[pos] - (9/11)*vy[pos-1] + (2/11)*vy[pos-2] + (6/11)*h*fn1_predicted
            elif order == 5:
                yn = (48/25)*vy[pos] - (36/25)*vy[pos-1] + (16/25)*vy[pos-2] - (3/25)*vy[pos-3] + (12/25)*h*fn1_predicted
            elif order == 6:
                yn = (300/137)*vy[pos] - (300/137)*vy[pos-1] + (200/137)*vy[pos-2] - (75/137)*vy[pos-3] + (12/137)*vy[pos-4] + (60/137)*h*fn1_predicted
            vy.append(yn)
            tn = tn + h
                

def main():
    read_file = open('entrada.txt', 'r')
    write_file = open('saida.txt', 'w')
    write_file.close()
    for line in read_file:
        input_str = line.split(' ')

        vt.clear()
        vy.clear()
        fn.clear()

        if input_str[0] == 'euler' or input_str[0] == 'euler_inverso' or input_str[0] == 'euler_aprimorado' or input_str[0] == 'runge_kutta':
            y0 = float(input_str[1])
            t0 = float(input_str[2])
            h = float(input_str[3])
            n = int(input_str[4])
            str_expr = input_str[5].strip('\n')
            expr = sympify(str_expr)
            if input_str[0] == 'euler':
                euler_method(y0, t0, h, n, expr)
                print_solution('Euler', h)
                plt.title('Metodo de Euler')
                method_str = 'Metodo de Euler'
            elif input_str[0] == 'euler_inverso':
                backward_euler_method(y0, t0, h, n, expr)
                print_solution('Euler Inverso', h)
                plt.title('Metodo de Euler Inverso')
                method_str = 'Metodo de Euler Inverso'
            elif input_str[0] == 'euler_aprimorado':
                improved_euler_method(y0, t0, h, n, expr)
                print_solution('Euler Aprimorado', h)
                plt.title('Metodo de Euler Aprimorado')
                method_str = 'Metodo de Euler Aprimorado'
            elif input_str[0] == 'runge_kutta':
                runge_kutta_method(y0, t0, h, n, expr)
                print_solution('Runge Kutta', h)
                plt.title('Metodo de Runge-Kutta')
                method_str = 'Metodo de Runge-Kutta'
            print_graph(method_str)
        elif input_str[0] == 'adam_bashforth':
            str_order = input_str[len(input_str)-1].split('\n')
            order = int(str_order[0])
            for i in range(1, order+1):
                vy.append(float(input_str[i]))
            y0 = vy[len(vy)-1]
            t0 = float(input_str[order+1])
            h = float(input_str[order+2])
            n = int(input_str[order+3])
            expr = sympify(input_str[order+4])
            for i in range(0, order):
                vt.append(t0)
                t0 = t0 + h
            adam_bashforth_method(y0, t0, h, n, expr, order)
            print_solution(('Adam Bashforth de ' + str(order) + ' ordem'), h) 
            plt.title('Adam Bashforth de ' + str(order) + ' ordem')
            method_str = 'Adam Bashforth de ' + str(order) + ' ordem'
            print_graph(method_str)
        elif input_str[0] == 'adam_multon':
            str_order = input_str[len(input_str)-1].split('\n')
            order = int(str_order[0])
            for i in range(1, order):
                vy.append(float(input_str[i]))
            y0 = vy[len(vy)-1]
            t0 = float(input_str[order])
            h = float(input_str[order+1])
            n = int(input_str[order+2])
            expr = sympify(input_str[order+3])
            for i in range(1, order):
                vt.append(t0)
                t0 = t0 + h
            adam_multon_method(y0, t0, h, n, expr, order)
            print_solution(('Adam Multon de ' + str(order) + ' ordem'), h)
            plt.title('Adam Multon de ' + str(order) + ' ordem')
            method_str = 'Adam Multon de ' + str(order) + ' ordem'
            print_graph(method_str)
        elif input_str[0] == 'formula_inversa':
            str_order = input_str[len(input_str)-1].split('\n')
            order = int(str_order[0])
            for i in range(1, order):
                vy.append(float(input_str[i]))
            y0 = vy[len(vy)-1]
            t0 = float(input_str[order])
            h = float(input_str[order+1])
            n = int(input_str[order+2])
            expr = sympify(input_str[order+3])
            for i in range(1, order):
                vt.append(t0)
                t0 = t0 + h
            backward_formula_method(y0, t0, h, n, expr, order)
            print_solution(('Formula Inversa de Diferenciacao de ' + str(order) + ' ordem'), h)
            plt.title('Formula Inversa de Diferenciacao de ' + str(order) + ' ordem')
            method_str = 'Formula Inversa de Diferenciacao de ' + str(order) + ' ordem'
            print_graph(method_str)
        elif input_str[0] == 'adam_bashforth_by_euler' or input_str[0] == 'adam_bashforth_by_euler_inverso' or input_str[0] == 'adam_bashforth_by_euler_aprimorado' or input_str[0] == 'adam_bashforth_by_runge_kutta':
            y0 = float(input_str[1])
            t0 = float(input_str[2])
            h = float(input_str[3])
            n = int(input_str[4])
            str_expr = input_str[5]
            expr = sympify(str_expr)
            order = int(input_str[6].strip('\n'))
            if input_str[0] == 'adam_bashforth_by_euler':
                euler_method(y0, t0, h, order-1, expr)
                method_str = 'Adam Bashforth de ' + str(order) + ' ordem por Euler'
            elif input_str[0] == 'adam_bashforth_by_euler_inverso':
                backward_euler_method(y0, t0, h, order-1, expr)
                method_str = 'Adam Bashforth de ' + str(order) + ' ordem por Euler Inverso'
            elif input_str[0] == 'adam_bashforth_by_euler_aprimorado':
                improved_euler_method(y0, t0, h, order-1, expr)
                method_str = 'Adam Bashforth de ' + str(order) + ' ordem por Euler Aprimorado'
            if input_str[0] == 'adam_bashforth_by_runge_kutta':
                runge_kutta_method(y0, t0, h, order-1, expr)
                method_str = 'Adam Bashforth de ' + str(order) + ' ordem por Runge-Kutta'
            y0 = vy[len(vy)-1] 
            t0 = vt[len(vt)-1] + h
            adam_bashforth_method(y0, t0, h, n, expr, order)
            print_solution(method_str, h)
            plt.title(method_str)
            print_graph(method_str)
        elif input_str[0] == 'adam_multon_by_euler' or input_str[0] == 'adam_multon_by_euler_inverso' or input_str[0] == 'adam_multon_by_euler_aprimorado' or input_str[0] == 'adam_multon_by_runge_kutta':
            y0 = float(input_str[1])
            t0 = float(input_str[2])
            h = float(input_str[3])
            n = int(input_str[4])
            str_expr = input_str[5]
            expr = sympify(str_expr)
            order = int(input_str[6].strip('\n'))
            if input_str[0] == 'adam_multon_by_euler':
                euler_method(y0, t0, h, order-2, expr)
                method_str = 'Adam Multon de ' + str(order) + ' ordem por Euler'
            elif input_str[0] == 'adam_multon_by_euler_inverso':
                backward_euler_method(y0, t0, h, order-2, expr)
                method_str = 'Adam Multon de ' + str(order) + ' ordem por Euler Inverso'
            elif input_str[0] == 'adam_multon_by_euler_aprimorado':
                improved_euler_method(y0, t0, h, order-2, expr)
                method_str = 'Adam Multon de ' + str(order) + ' ordem por Euler Aprimorado'
            if input_str[0] == 'adam_multon_by_runge_kutta':
                runge_kutta_method(y0, t0, h, order-2, expr)
                method_str = 'Adam Multon de ' + str(order) + ' ordem por Runge-Kutta'
            y0 = vy[len(vy)-1] 
            t0 = vt[len(vt)-1] + h
            adam_multon_method(y0, t0, h, n, expr, order)
            print_solution(method_str, h)
            plt.title(method_str)
            print_graph(method_str)
        elif input_str[0] == 'formula_inversa_by_euler' or input_str[0] == 'formula_inversa_by_euler_inverso' or input_str[0] == 'formula_inversa_by_euler_aprimorado' or input_str[0] == 'formula_inversa_by_runge_kutta':
            y0 = float(input_str[1])
            t0 = float(input_str[2])
            h = float(input_str[3])
            n = int(input_str[4])
            str_expr = input_str[5]
            expr = sympify(str_expr)
            order = int(input_str[6].strip('\n'))
            if input_str[0] == 'formula_inversa_by_euler':
                euler_method(y0, t0, h, order-2, expr)
                method_str = 'Formula Inversa de Diferenciacao de ' + str(order) + ' ordem por Euler'
            elif input_str[0] == 'formula_inversa_by_euler_inverso':
                backward_euler_method(y0, t0, h, order-2, expr)
                method_str = 'Formula Inversa de Diferenciacao de ' + str(order) + ' ordem por Euler Inverso'
            elif input_str[0] == 'formula_inversa_by_euler_aprimorado':
                improved_euler_method(y0, t0, h, order-2, expr)
                method_str = 'Formula Inversa de Diferenciacao de ' + str(order) + ' ordem por Euler Aprimorado'
            if input_str[0] == 'formula_inversa_by_runge_kutta':
                runge_kutta_method(y0, t0, h, order-2, expr)
                method_str = 'Formula Inversa de Diferenciacao de ' + str(order) + ' ordem por Runge-Kutta'
            y0 = vy[len(vy)-1] 
            t0 = vt[len(vt)-1] + h
            backward_formula_method(y0, t0, h, n, expr, order)
            print_solution(method_str, h)
            plt.title(method_str)
            print_graph(method_str)
        pc.append(vy[500])
    read_file.close()
    f = open('comparison.txt', 'w')
    for i in pc:
        f.write(str(i) + '\n')
    f.close()

if __name__ == "__main__":
    main()