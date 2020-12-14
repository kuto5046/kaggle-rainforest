# kaggle-rainforest
<div align="center"><img src="./img/001.png" title="result ε scheduling"></div>
https://www.kaggle.com/c/rfcx-species-audio-detection/overview 

## [2020/12/12]
コンペ取り組み開始  
アライさんのnotebookをひとまず読む
https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english　

### 音声向けのDA
| 手法 | 説明 |  特徴  |
| --- | --- | --- |
| GaussianNoise(white noise) |  正規分布に従うノイズ | ノイズの強さを指定して与えるので音声が小さい時ノイズで覆ってしまう |
| GaussianNoise(SNR) | SNRを考慮して適応的にノイズを与える | 元の音量はばらつきがあるので基本的にSNRを考慮したほうがいい |
| PinkNoise | 低周波数帯から徐々にノイズの強さを減少するようなノイズ | 自然界に存在するノイズはこれに近い  |
| PitchShift | 音のピッチ(高低)に関する調整 | 聞こえる音が高く/低くなる(スペクトログラムの場合パターンが上下する) 時間がかかる 音割れに注意 | 
| TimeStretch | 音を時間的に伸ばしたり縮めたりする | 時間がかかる | 
| TimeShift | 音を時間的にずらす  | ずらした余りの部分は削除するか前に持ってくるか |
| VolumeControl | 音量そのものを調整する | メルスペクトログラムに影響を与える(種類もいくつかある) |

SNR = SignalとNoiseの比   
ex)SNR > -1 -> Signalが大きい 

### データ形式
tfrecordsはどういう意義がある？
flac形式だけど変換はどうする？
birdcallではmp3 → wav(その際にサンプリング周波数も揃えていた)

wav:オリジナルデータ
mp3:非可逆圧縮形式
flac:可逆圧縮形式

flac → mp3に変換することで軽量化する

### データについて
trainのcsvは２種類存在する
|    |    |    |
| --- | --- | --- |
| train_fp.csv | false positiveデータ(鳴いていないんだけど鳴いているとラベルつけしているデータ)|  7781 |
| train_tp.csv | true positiveデータ(通常の正解データ) | | 

baselineとしてはtpデータのみでやってみる
訓練用のメタデータには時間ラベルも振ってあるが予測は音声ファイル単位で良い

#### fpデータはどのように使えそう？
fpはAという鳥がいないのにいるといっている状態  
これをそのまま使ってしまうと２つの意味でノイズとなってしまう  
1. 鳥Bの鳴き声を鳥Aと学習してしまうので鳥Aの予測においてノイズとなる  
2. 鳥Bが鳴いているのにラベルが付与されていない？場合は鳥Bの予測にもノイズとなる  
2はどうなんだろう？  


