from fusion.models.fusion import fusion
from fusion.d2_reader import detection_2d_reader

fus = fusion()
print(fus.name)
ret = detection_2d_reader(10)
print(ret)