
import numpy as np
import pandas as pd

working_folder = './TM40_46prod_Mauro_SingleEye/'
batch = 4

n_exec = 1
n_fold = 10

all_med = []
all_median = []

for i in range(n_fold):
    folder_name = working_folder + 'Exec_%i/'%(n_exec)
    filename = folder_name + '/outputs_prod/dice_metric_production_%s'%(batch) + '.txt'
    with open(filename, 'r') as f:
        data = list(np.loadtxt(f, dtype='str', delimiter=', '))
    data[0] = data[0][1:]
    data[-1] = data[-1][:-1]
    data = np.float_(data)

    media = np.mean(data)
    mediana = np.median(data)
    desvio = np.std(data)

    all_med.append(media)
    all_median.append(mediana)

    print(media, mediana, desvio)

    d = {
        'Media': [media],
        'Mediana': [mediana],
        'Desvio Padrao': [desvio]
    }
    df = pd.DataFrame(data=d)

    df.to_csv(folder_name + 'data_table_production_%s.csv'%(batch), index=False)
    n_exec += 1

#all_med.pop(6)
#all_median.pop(6)

with open(working_folder + 'mean_median_results.txt', 'w') as f:
    f.write("Médias: " + str(all_med))
    f.write('\n')
    f.write("Média das Médias: " + str(np.mean(all_med)))
    f.write('\n')
    f.write("Medianas: " + str(all_median))
    f.write('\n')
    f.write("Média das Medianas: " + str(np.mean(all_median)))
    f.write('\n')
    f.write('\n')
    f.write("Mínimo das Medias: " + str(np.min(all_med)))
    f.write('\n')
    f.write("Máximo das Medias: " + str(np.max(all_med)))