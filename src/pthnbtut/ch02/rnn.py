#coding:UTF-8
'''
Created on 2019年6月16日-上午11:50:28
author: Gary-W
'''

import torch
torch.__version__

rnn = torch.nn.RNN(20,50,2)
input = torch.randn(100 , 32 , 20)
h_0 =torch.randn(2 , 32 , 50)
output,hn=rnn(input ,h_0) 
print(output.size(),hn.size())


if __name__=="__main__":
    pass

