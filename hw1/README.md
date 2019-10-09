# Requirements

```
numpy==1.16.4
opencv3==3.1.0
```

## eval.py
執行下面指令會讀取原圖跟BF和JBF的GT，用來驗證手刻JBF的正確性
理應print出"0 0"
```
python eval.py
```

## Advanced_RGB2Gray.py
讀取一張照片後，會輸出三張weight組合投票分數最高的灰階圖

```
python Advanced_RGB2Gray.py --input_path your_image_path
```
