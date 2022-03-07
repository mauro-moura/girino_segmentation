Outputs
- Os nomes dos GTs foram alterados de acordo com a slice correspondente (meramente comparativo)

main.py
- Foi adicionada uma verificação se as imagens e os GTs estão sendo importados na mesma ordem
- size_img agora inicia com o valor do tamanho da imagem, porém com o tempo é atualizado automaticamente de acordo com o novo tamanho das imagens
- O tamanho definido(chute inicial) foi de 256x256. O dice obtido com esses para cada volue conhecido pode ser observado na pasta de outputs
- No momento em que as imagens são salvas, agora é definido também um novo tamanho para que sejam salvas com o mesmo tamanho das originais, ou seja o reverse_size
- As imagens salvas agora possuem o mesmo número das imagens de entrada

unet.py
- Um novo modelo de Unet foi adicionado para fins de teste e comparação(até mesmo para ter uma execução mais leve), porém não será utilizado no momento.

utils.py
- Algumas funções foram adicionadas no método load_images
- Foi adicionada a função get_shape_resize para automaticamente pegar um novo tamanho mais próximo do original para as imagens, porém essa não é utilizada já que o tamanho das imagens é grande demais para a GPU
- Foi adicionada a função resize_img para alteração do tamanho das imagens de acordo com um novo tamanho definido
- Foi adicionada a função reverse_size para que as imagens sejam salvas em seu tamanho original

data_build.py
- Foi descomentado o if else para GTs
- Foi alterado para que as GTs possuam o mesmo número das imagens de treinamento
