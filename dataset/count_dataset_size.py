import os
# train:12880
# val:3226
path = "./WIDER_val/images"

def count(path):
    if os.path.isfile(path):
        return 1
    else:
        sum = 0
        for file in os.listdir(path):
            sum += count(os.path.join(path,file))
        return sum

print(count(path))
