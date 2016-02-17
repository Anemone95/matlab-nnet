% Author:         Anemone
% Filename:       demo.m
% Last modified:  2016-02-16 21:00
% E-mail:         x565178035@126.com

x1=[1:3:20]';
x2=[1:3:20]';
x=[x1,x2];

y=x1.^2+x2.^2;

[net,is,os]=EzBP(x,y);

testNum=[11,11]';

%输入值归一化
inputNum=mapminmax('apply',testNum,is);
%放入神经网络,进行计算
outputNum=net(inputNum);
%将神经网络的值反归一化
res=mapminmax('reverse',outputNum,os)
