<b> RNN Application</n>

Recurrent Neural Network</n>
    We can precess a sequence of vectors x by applying a recurrence</n>
    formula at every time step:</n>
    ht = fw(ht-1, xt)</n>
    ht : new state</n>
    fw : some function with parameters W</n>
    ht-1 : old state</n>
    xt : input vector at some time step</n>
    output is connected to cell, then it will be input.</n>

    ht(1,5,2) : (batch_size, sequence_length ,hidden_size)
       ↑
    xt(1,5,4) : (batch_size, sequence_length, input_dimension)

for Example</n>
    Language Modeling</n>
    Speech Recognition</n>
    Machine Translation</n>
    Conversation Modeling/Question Answering</n>
    Image/Video Captioning</n>
    Image/Music/Dance Generation</n>
