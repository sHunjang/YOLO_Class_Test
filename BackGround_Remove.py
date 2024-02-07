from rembg import remove
from PIL import Image



# 이미지 배경 제거
input = Image.open('path/to/remove/img.png') # load image
output = remove(input) # remove background
output.save('save/img/name.png') # save image