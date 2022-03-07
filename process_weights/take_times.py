
working_folder = './TM40_46prod_Mauro_SingleEye/'

tempos = []

n_exec = 1
n_fold = 10

for i in range(n_fold):
    with open(working_folder + 'Exec_%i/tempos_4_100_exec_%i.txt'%(n_exec, n_exec), 'r') as f:
        text = f.read()
    print(text)
    a = text.split()
    tempos.append(round(float(a[-1][:-1]), 3))
    print("Tempo de execução: %.3f"%(tempos[-1]))
    n_exec += 1

with open(working_folder + 'Times.txt', 'w') as f:
    f.write(str(tempos))
