convolution neural network

입력을 나누어서 받음
conv
relu
pool
fully connected layer(FC)

output : 전체 사이즈(N*N)
    output size = {(N - F)/stride}+1
    ex) In case of total 7*7 and filter 3, stride 1
        {(7 - 3)/1}+1 = 5
        therefore, output size is 5*5
        if stride is 3, {(7-3)/3}+1 = 2.333333. then in this case, we can't define stride 3.
        so you should check the possible stride value.
filter : 한번에 뽑을 칸의 갯수 (F*F)
stride : 몇칸씩 움직일건지의 단위

Padding
    the bigger you input stride, the more you loss the data.
    and the pixels on the corners and the edges are used much less than those in the middle.
    padding : filter를 적용하여 학습시키면 계속해서 이미지의 크기가 줄어듬. 그래서 padding이라는 0의 값으로 채워줌
    please refer to this site. https://www.geeksforgeeks.org/cnn-introduction-to-padding/
    With "SAME" padding, if you use a stride of 1, the layer's outputs will have the same spatial dimensions as its inputs.
    With "VALID" padding, there's no "made-up" padding inputs. The layer only uses valid input data.

Convolution Layer
    step 1 : 32*32*3 image, padding 0, stride 1
             conv, ReLU = e.g.6 5*5*3 filters
    step 2 : 28*28*6 image, padding 0, stride 1
             conv, ReLU = e.g.10 5*5*6 filters
    step 3 : 24*24*10 image, padding 0, stride 1

Pooling Layer(sampling)
    1. pick up one layer of convolution layer
    2. resize(sampling) of one layer picked up No.1
    3. repeat that No.1 and No.2 and laminate samplings
Max Pooling : select the maximum number of one layer picked up

fully connected layer(FC) : Contains neurons that connect to the entire input volume, as in ordinary Neural Network
                            Collection of conv, ReLU, Pool