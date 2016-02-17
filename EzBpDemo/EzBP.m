function [net,ps,ts]=EzBP(P,T,x);
    % input x个体初始权值和阀值
    % input P样本输入(n line,1 col)
    % input T样本输出(n line,1 col)

    % output net BP神经网络
    % output ps  输入值归一化矩阵
    % output ts  输出值归一化矩阵

    nntwarn off     %关闭警告
    P=P';
    T=T';
    %数据预处理--归一化处理
    [P,ps]=mapminmax(P);
    [T,ts]=mapminmax(T);
    [pr,pc]=size(P);
    [tr,tc]=size(T);

    %设置隐藏神经元个数,一般设置2*inputNum+1
    inputNum=pr;
    outputNum=tr;
    hiddenNum=2*inputNum+1;

    %这里采用tansig的激励算子,这个算子对非线性的插值计算效果较好
    net=newff(minmax(P),[hiddenNum,outputNum],{'tansig','tansig'}); %隐含层 输出层

    %设置神经网络训练的结束条件
    net.trainParam.epochs=1e5;
    net.trainParam.goal=1e-5;
    net.trainParam.lr=0.05;
    net.trainParam.show=10;

    if nargin==3
        net.trainParam.showwindow=false;
        w1num=inputNum*hiddenNum;
        w2num=outputNum*hiddenNum;
        w1=x(1:w1num);
        B1=x(w1num+1:w1num+hiddenNum);
        w2=x(w1num+hiddenNum+1:w1num+hiddenNum+w2num);
        B2=x(w1num+hiddenNum+w2num+1:w1num+hiddenNum+w2num+outputNum);
        net.iw{1,1}=reshape(w1,hiddenNum,inputNum);
        net.lw{2,1}=reshape(w2,outputNum,hiddenNum);
        net.b{1}=reshape(B1,hiddenNum,1);
        net.b{2}=reshape(B2,outputNum,1);
    end
    %训练
    net=train(net,P,T);
end
