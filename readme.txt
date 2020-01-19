development tool : IntelliJ Pycharm
        PyCharm 2019.2.3 (Community Edition)
        Build #PC-192.6817.19, built on September 25, 2019
        Runtime version: 11.0.4+10-b304.69 x86_64
        VM: OpenJDK 64-Bit Server VM by JetBrains s.r.o
        macOS 10.15.2
        GC: ParNew, ConcurrentMarkSweep
        Memory: 990M
        Cores: 4
        Registry: 
        Non-Bundled Plugins: com.alayouni.ansiHighlight, com.andrey4623.rainbowcsv, com.intellij.ideolog, net.seesharpsoft.intellij.plugins.csv
Language : Python 3.6.9 :: Anaconda, Inc.
library 
    - tensorflow 1.9
    - numpy 1.16.5
    - matplotlib 3.1.1

Hypothesis : 가설에 해당하는 함수(ex. H(x) = Wx, W: 기울기)
cost : 실제 데이터와 가설의 데이터 사이의 차이가 얼마나 나느지 측정한 것(실제값과 가설값의 차이의 평균)
Gradient descent(W) : cost의 기울기 -> 0으로 수렴할때까지

reference
CPU AVX AVX2 problem
    https://blog.naver.com/wjddn9252/221587230559

Convex function -> hypothesis를 이용하면 항상 값을 찾을 수 있도록 특정 지점으로 수렴

install of numpy version 1.16.5
    https://antilibrary.org/2259
install of tensorflow version less than 2.0

