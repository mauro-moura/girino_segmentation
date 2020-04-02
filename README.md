# Girino Segmentation v0.01b

Esse é um repositório de segmentação de imagens de girinos.

A versão do python utilizada foi a 3.6, juntamente com o tensorflow 1.15. (No momento não funciona com tensorflow 2). os demais requisitos constam no arquivo requirements.txt

Nesse caso para instalação dos requerimentos só é preciso a execução do comando "pip install -r requirements.txt"

O arquivo principal para execução é o main.py, basta executá-lo e as pastas necessárias serão conferidas e criados se necessário. Os arquivos que foram utilizados na Augmentation ficarão salvos na pasta ./dados_girino/Aug_Train e ./dados_girino/Aug_GT os resultados preditos da Rede Neural estarão disponíveis na pasta ./outputs

Outros arquivos:

    * utils.py contém funções de utilidade

    * unet.py contém o modelo da unet e as métricas

Caso queira executar no colab, as instruções básicas são:

    * O arquivo principal, e único que precisa ser aberto no colab é o main_colab.ipynb

    * A pasta definida como principal é ./Labic Segmentation/girino_segmentation, caso decida salvar em uma pasta diferente é necessário mudar

    * Verifique se seu Drive está montado (caso não esteja haverão instruções no notebook para isso)

    * Verifique se o seu runtime está habilitado para executar na GPU, basta clicar na aba Runtime > Change Runtime Type > Hardware accelerator. Nesse caso a opção deve ser GPU.

    * Basta executar todas as células.
