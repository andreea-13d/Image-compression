# Nume student: 
####################################
# DVS: Compresia de imagini alb negru
####################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from datetime import datetime


def my_algorithm():
        # algorithm code here
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

            
    #Cititi o imagine rgb la alagere
    Img =  mpimg.imread('LOGO_3000.png') # remarca: matplotlib accepta doar imagini .png
    fig1 =plt.figure(1)
    plt.imshow(Img)

    fig2 =plt.figure(2)
    grayImg = rgb2gray(Img)
    plt.imshow(grayImg, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

    # plt.show()
    #1. Aplicati svd pe imaginea gray 



    def dvs(grayImg):
        [u,s,v] = np.linalg.svd(grayImg)
        return [u,s,v]

    # print("-------------S------------")
    # print(s)
    # print("-------------V--------------")
    # print(v)
    start = datetime.now()
    [u, s, v]= dvs(grayImg)





    #2. Plotati valorile singulare pe scala logaritmica utilizand plt.semilogy

    # plt.semilogy(s)
    # plt.show()

    #    Adaugati comentariu cu ce observati

    #3. Plotati graficul procent informatie vs valori singulare
    #Hint: utilizati functia np.cumsum.
    #Adaugati comentariu cu ce observati

    suma = np.cumsum(s)/np.sum(s)
    # plt.plot(suma)
    # plt.show()

    #90% - 270 valori
    #60% - 50 valori
    #50% - 20 valori
    #30% - 7 valori
    #20% - 3 valori


    #4. In urma analizei graficului procent informatie vs valori singulare generati
    #un vector de dimensiune 5

    nr_val = [270 , 50 , 20 , 7 , 3]


    #5. Utilizati elementele vectorului pentru a reconstrui imaginile

    # for i in nr_val:
    #     imag_noua = u[:,:i] @ np.diag(s[:i]) @ v[:i,:]
    #     plt.imshow(imag_noua, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    #     plt.show()

    imagine_salvata = u[:,:250] @ np.diag(s[:250]) @ v[:250,:]
    stop = datetime.now()

    print(f"Durata {stop-start}")


my_algorithm()
    # plt.imsave('zaruri_dvs.png', imagine_salvata, cmap='gray')


#Toate imaginile au fost reconstituite la 95% din imaginea initiala

#Nr pixeli            Timp
# 64x64              0.007 sec
# 100x100            0.012 sec
# 500x500            0.31 sec
# 1000x1000          1.3 sec
# 1500x1500          3.45 sec
# 1500x1500          3.45 sec
# 2000x2000          7.2 sec
# 3000x3000          7.2 sec
# 4000x4000          7.2 sec
# 5000x5000          7.2 sec
# 6000x6000          7.2 sec
# 7000x7000          7.2 sec
# 8000x8000          7.2 sec