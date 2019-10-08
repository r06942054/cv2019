import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):
        
        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
    
    def rgb2gray(self, img, weight):
        return (weight[0] * img[:,:,0] + weight[1] * img[:,:,1] + weight[2] * img[:,:,2])
    
    def joint_bilateral_filter(self, input, guidance):
        
        r = 3 * self.sigma_s  # default: 9
        ws = 2 * r + 1        # default: 19
        
        # Padding
        img_pad = cv2.copyMakeBorder(input, r, r, r, r ,cv2.BORDER_REFLECT)
        guidance_pad = cv2.copyMakeBorder(guidance, r, r, r, r ,cv2.BORDER_REFLECT)
        
        # new array for saving filtered image
        output = np.zeros((input.shape[0], input.shape[1], 3))
        
        # calculate Spatial kernel (default: 19*19)
        Gs = np.zeros((ws, ws, 3))
        for i in range(ws):
            for j in range(ws):
                Gs[i,j, :] = (-1) * ((i-r)**2 + (j-r)**2) / (2 * (self.sigma_s**2))
        Gs = np.exp(Gs)
        
        # Apply the JBF
        # 計算每個target point時，會先找出kernal size的array來做運算 (default: 19*19)
        for x in range(input.shape[0]):
            for y in range(input.shape[1]):
                Iq_arr = img_pad[x:x+ws, y:y+ws]
                
                # calculate Range Kernel (default: 19*19)
                Tq_arr = guidance_pad[x:x+ws, y:y+ws] / 255 # normalize before calculating Range Kernel
                Tp = guidance_pad[x+r, y+r] / 255           # normalize before calculating Range Kernel
                if len(Tq_arr.shape)==3: # for RGB image
                    Gr = (-1) * np.sum(np.square(Tq_arr - Tp), axis=2) / (2 * (self.sigma_r**2))
                else: # for Gray image
                    Gr = (-1) * np.square(Tq_arr - Tp) / (2 * (self.sigma_r**2))
                Gr = np.exp(Gr)
                Gr = np.stack((Gr, Gr, Gr), axis=-1)
                
                output[x, y] = np.sum(np.sum((Gs * Gr * Iq_arr), axis=0), axis=0) / np.sum(np.sum((Gs * Gr), axis=0), axis=0)
                
        return output


