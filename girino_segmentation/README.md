[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mauro-moura/girino_segmentation/blob/master/main_colab.ipynb)

# Girino Segmentation v1.0

## Tabela de Conteúdos

[Introdução](https://github.com/mauro-moura/girino_segmentation#Introdução) </br>
[Arquivos](https://github.com/mauro-moura/girino_segmentation#Arquivos-Explicados) </br>

Para ver as atualizações basta [clicar aqui](https://github.com/mauro-moura/girino_segmentation/updates.md)


Estrutura das pastas:

```
dados_girino

    A1_norm_images -> Pasta para imagens de entrada processadas
    A2_GT_images   -> Pasta para imagens de ground truth processadas
    GT_Producao    -> Imagens de Ground Truth utilizadas para teste em produção da rede
    Producao       -> Imagens utilizadas para teste em produção da rede


data_build/dados_girino

    A0_Avizo_images    -> Imagens de Ground Truth obtidas manualmente
    A0_filtered_images -> Imagens de entrada
    A1_norm_images     -> Pasta para imagens de entrada processadas
    A2_GT_images       -> Pasta para imagens de ground truth processadas
```

Para o tensorflow 1.15 é utilizado CUDA 10.0 com CUDNN v7.6

# Introdução

Esse é um repositório de segmentação de imagens de girinos.

A versão do python utilizada foi a 3.6, juntamente com o tensorflow 1.15. (No momento não funciona com tensorflow 2). os demais requisitos constam no arquivo requirements.txt

Nesse caso para instalação dos requerimentos só é preciso a execução do comando "pip install -r requirements.txt"

O arquivo principal para execução é o main.py, basta executá-lo e as pastas necessárias serão conferidas e criados se necessário. É possível habilitar para que os arquivos que foram utilizados na Augmentation fiquem salvos na pasta ./dados_girino/Aug_Train e ./dados_girino/Aug_GT, porém por gerar diversas imagens acaba gerando muitos arquivos.

A função que faz a execução do treinamento da rede é a model.fit_generator, o parâmetro steps_per_epoch é responsável pelo número de imagens que será usado no treinamento (a partir da augmentation) e o epochs é responsável pelo número de loops.

Após a execução do código, todos os arquivos serão armazenados na pasta outputs, dentre eles um arquivo contendo o tempo de execução do treinamento do modelo, tempo de predição das imaens e tempo total de execução.


# Arquivos Explicados

Outros arquivos:

    * utils.py contém funções de utilidade

    * unet.py contém o modelo da unet e as métricas
	
	* data_build.py é uma função usada para conversão das imagens coloridas para binário

Caso queira executar no colab, as instruções básicas são:

    * O arquivo principal, e único que precisa ser aberto no colab é o main_colab.ipynb

    * A pasta definida como principal é ./Labic Segmentation/girino_segmentation, caso decida salvar em uma pasta diferente é necessário mudar

    * Verifique se seu Drive está montado (caso não esteja haverão instruções no notebook para isso)

    * Verifique se o seu runtime está habilitado para executar na GPU, basta clicar na aba Runtime > Change Runtime Type > Hardware accelerator. Nesse caso a opção deve ser GPU.

    * Basta executar todas as células.
