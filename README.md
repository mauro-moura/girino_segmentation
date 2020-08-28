# Girino Segmentation v1.0

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mauro-moura/girino_segmentation/blob/master/main_colab.ipynb)

## Sobre

Esse é um repositório de segmentação de imagens de girinos.

## Requisitos
A versão do python utilizada foi a 3.6, juntamente com o tensorflow 1.15. (No momento não funciona com tensorflow 2). os demais requisitos constam no arquivo requirements.txt

Nesse caso para instalação dos requerimentos só é preciso a execução do comando 
```bash
pip install -r requirements.txt
```

## Estrutura de arquivos

O arquivo principal para execução é o main.py, basta executá-lo e as pastas necessárias serão conferidas e criados se necessário. É possível habilitar para que os arquivos que foram utilizados na Augmentation fiquem salvos na pasta ./dados_girino/Aug_Train e ./dados_girino/Aug_GT, porém por gerar diversas imagens acaba gerando muitos arquivos.

A função que faz a execução do treinamento da rede é a model.fit_generator, o parâmetro steps_per_epoch é responsável pelo número de imagens que será usado no treinamento (a partir da augmentation) e o epochs é responsável pelo número de loops.

Após a execução do código, todos os arquivos serão armazenados na pasta outputs, dentre eles um arquivo contendo o tempo de execução do treinamento do modelo, tempo de predição das imaens e tempo total de execução.

Outros arquivos:

* utils.py contém funções de utilidade

* unet.py contém o modelo da unet e as métricas

* data_build.py é uma função usada para conversão das imagens coloridas para binário

## Execução no Google Colab

Caso queira executar no colab, as instruções básicas são:

* O arquivo principal, e único que precisa ser aberto no colab é o `main_colab.ipynb`
* A pasta definida como principal é **./Labic Segmentation/girino_segmentation**, caso decida salvar em uma pasta diferente é necessário alterar no código.
* Verifique se seu Drive está montado (caso não esteja haverão instruções no notebook para isso).
* Verifique se o seu runtime está habilitado para executar na GPU, basta clicar na aba :
> Runtime > Change Runtime Type > Hardware accelerator.
Nesse caso a opção deve ser GPU.
