# cv2019
NTU course Computer Vision 2019FALL

## HW1

投影片: http://media.ee.ntu.edu.tw/courses/cv/19F/hw/hw1.pdf

Bilateral filter有去除noise並保留邊緣的效果 (參見cv2019_lec02投影片84頁)

* Problem:
**將RGB彩色圖片轉成灰階時，會造成失真**
因為轉成灰階是使用 Y = Wr * R + Wg * G + Wb * B進行降維，若是在同一個平面上的色彩，會轉到相同的灰階值，導致在灰階圖上無法用肉眼分辨同一平面的色彩
既然同一個平面上的色彩都會有失真，那對每一張照片去找出一組Wr、Wg、Wb並讓失真最小，會是一個較好的solution
    
* 作法

先將Weight的組合做quantize，總共66種組合 (H3取10=66)
![](https://i.imgur.com/eozMLin.png)

1. 將原圖經過Bilateral filter (BF)

2. 將原圖經過某一組Wr、Wg、Wb轉成灰階後當作Joint Bilateral filter (JBF)的Guidance後與原圖去做JBF
3. 計算1和2之間的L1 Norm
4. 將66種組合的cost在二維平面上找local minimum (附近6個點，邊界的點附近2~4點)，我使用的方式是找出L1 distance=0.2的點，去找局部最小值，只要是局部最小值，則該組weight投一票
5. sigma_r跟sigma_s總共九種組合，都跑過1~4的流程後，輸出前三張票數最高的灰階照片


* JBF公式:
原作者的paper: https://ybsong00.github.io/siga13tb/siga13tb_final.pdf?fbclid=IwAR10r4ARhDw3oCC6FtWMe7moHI0WlN1CK4S2wzyrYQxvDpfY5qmQObktDYI
![](https://i.imgur.com/n8lgSgl.png)
![](https://i.imgur.com/6ZDGWA5.png)

p是filter的中心點index，q是目標點index，Omega_p是filter的所有點index的集合
Gs是Spatial Kernel，Gr是Range Kernel
Tp是Guaidence的中心點的value
Tq是Guaidence的目標點的value


若Guidence是灰階(1個channel)，則複製拓展成3個channel去計算較快(都是numpy array直接矩陣計算，不用for迴圈去計算)


* 結果

1. testdata/1a.png

[9. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 9.]

[Wr, Wg, Wb]

Top 1 weights:  [1.0, 0.0, 0.0]

Top 2 weights:  [0.0, 0.0, 1.0]

Top 3 weights:  [0.0, 1.0, 0.0]



2. testdata/1b.png

[9. 0. 0. 0. 0. 0. 0. 0. 0. 0. 4. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]


[Wr, Wg, Wb]

Top 1 weights:  [0.0, 0.0, 1.0]

Top 2 weights:  [0.0, 1.0, 0.0]

Top 3 weights:  [1.0, 0.0, 0.0]



3. testdata/1c.png

[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 9. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 9.]

[Wr, Wg, Wb]

Top 1 weights:  [1.0, 0.0, 0.0]

Top 2 weights:  [0.0, 1.0, 0.0]

Top 3 weights:  [0.3, 0.7, 0.0]

1c.png與硯澤結果不同，有可能是因為他在轉gray有先四捨五入，我沒有
