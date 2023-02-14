import matplotlib.pyplot as plt
import numpy as np

def showfeature(input,picname):
    # plt.imshow(np.transpose(input[0],(1,2,0)).detach().cpu().numpy())
    # plt.show()
    print(input.max())
    plt.figure(figsize=(8*2,8*2))
    cnt = 0
    for j in range(input.size()[1]):
        cnt = cnt + 1
        plt.subplot(input.size()[1]//8,8,cnt)
        plt.imshow(input[0][cnt-1].detach().cpu().numpy()) #,cmap='gray'
    plt.savefig(picname+'.png')


def calculate_distance(x, y):
    # import numpy as np
    x = np.array(x)
    y = np.array(y)
    CosineDistance = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return CosineDistance


