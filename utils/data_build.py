
import glob
from PIL import Image

x = sorted(glob.glob('./dados_girino/Normalized_Teste_20slices/*.tif'))
output = './dados_girino/NEW_GT/'

for i in range(len(x)):
    temp = Image.open(x[i])
    temp = temp.convert('L')
    temp = temp.point(lambda x: 255 if x<196 else 0, '1')
    temp.save(output + "GT_bin0%i.tif"%(x[i][-8:-4]), "TIFF")
