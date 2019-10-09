import numpy as np
import cv2
import argparse
import time
import os

from joint_bilateral_filter import Joint_bilateral_filter

# find the best weights for RGB2Gray

def main():
    parser = argparse.ArgumentParser(description='JBF evaluation')
    parser.add_argument('--input_path', default='./testdata/ex.png', help='path of input image')

    args = parser.parse_args()
    
    img = cv2.imread(args.input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    guidance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vote = np.zeros(66)
    
    # 產生66種weight的組合 (wr, wg, wb)
    # wr + wg + wb = 1
    # wr, wg, wb in set {0, 0.1, 0.2, ..., 1.0}
    weight_table = []
    for i in range(0, 11):
        for j in range(0, 11):
            for k in range(0, 11):
                if (i+j+k) == 10:
                    weight_table.append([np.around(i/10, 1), np.around(j/10, 1), np.around(k/10, 1)])
    #print(*weight_table)
    print("len(weight_table): ", len(weight_table)) # it should print 66

    for sigma_s in [1, 2, 3]:
        for sigma_r in [0.05, 0.1, 0.2]:
            JBF = Joint_bilateral_filter(sigma_s, sigma_r, border_type='reflect')
            bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
            cost = []
            
            # calculate JBF: 66 combinations 
            for weight in weight_table:
                guidance = JBF.rgb2gray(img_rgb, weight).astype(np.uint8)
                jbf_out = JBF.joint_bilateral_filter(img_rgb, guidance).astype(np.uint8)
                
                cost_image = np.abs(bf_out-jbf_out)
                cost.append(np.sum(cost_image))
                
                print(sigma_s, sigma_r, weight, np.sum(cost_image))
                
            # voting
            for i in range(len(weight_table)):
                # find neighbors's cost (在2D平面wr+wg+wb=1上找local minimum)
                neighbor_cost = []
                
                for j in range(len(weight_table)):
                    
                    # 附近的點L1距離應為0.2，例如 (1, 0, 0) 和 (0.9, 0.1, 0)
                    if np.around(np.sum(np.abs(np.array(weight_table[i]) - np.array(weight_table[j]))), 1) == 0.2: # 加起來會變0.19999這狀況，需要做around
                        neighbor_cost.append(cost[j])
                
                #若比附近的點還要小，則是local minimum
                if cost[i] < min(neighbor_cost):
                    vote[i] += 1
                    
            print(vote)

    print(vote.argsort()[-3:][::-1]) #回傳vote前三名的index
    index = vote.argsort()[-3:][::-1]
    
    img_name = os.path.basename(args.input_path).split('.')[0]
    
    print("Top 1 weights: ", weight_table[index[0]])
    guidance = JBF.rgb2gray(img, weight_table[index[0]]).astype(np.uint8)
    cv2.imwrite(img_name + "_y1.png", guidance)
    
    print("Top 2 weights: ", weight_table[index[1]])
    guidance = JBF.rgb2gray(img, weight_table[index[1]]).astype(np.uint8)
    cv2.imwrite(img_name + "_y2.png", guidance)
    
    print("Top 3 weights: ", weight_table[index[2]])
    guidance = JBF.rgb2gray(img, weight_table[index[2]]).astype(np.uint8)
    cv2.imwrite(img_name + "_y3.png", guidance)
    
if __name__ == '__main__':
    main()
