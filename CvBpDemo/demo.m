% Author:         Anemone
% Filename:       demo.m
% Last modified:  2016-02-16 21:00
% E-mail:         x565178035@126.com

x1=[1:3:20]';
x2=[1:3:20]';
x=[x1,x2];

y=x1.^2+x2.^2;

net=CvBP(x,y,10);

testNum=[11,11]';

%放入神经网络,进行计算
outputNum=net(testNum)
