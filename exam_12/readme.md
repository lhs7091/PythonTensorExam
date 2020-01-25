<b> RNN Application

Recurrent Neural Network
    We can precess a sequence of vectors x by applying a recurrence
    formula at every time step:
    ht = fw(ht-1, xt)
    ht : new state
    fw : some function with parameters W
    ht-1 : old state
    xt : input vector at some time step

    output is connected to cell, then it will be input.

    ht(1,5,2) : (batch_size, sequence_length ,hidden_size)
       ↑
    xt(1,5,4) : (batch_size, sequence_length, input_dimension)

for Example
    Language Modeling
    Speech Recognition
    Machine Translation
    Conversation Modeling/Question Answering
    Image/Video Captioning
    Image/Music/Dance Generation
