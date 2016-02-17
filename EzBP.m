function [net,ps,ts]=EzBP(P,T,x);
    % input x�����ʼȨֵ�ͷ�ֵ
    % input P��������(n line,1 col)
    % input T�������(n line,1 col)

    % output net BP������
    % output ps  ����ֵ��һ������
    % output ts  ���ֵ��һ������

    nntwarn off
    P=P';
    T=T';
    [P,ps]=mapminmax(P);
    [T,ts]=mapminmax(T);
    [pr,pc]=size(P);
    [tr,tc]=size(T);


    inputNum=pr;
    outputNum=tr;
    hiddenNum=2*inputNum+1;
    net=newff(minmax(P),[hiddenNum,outputNum],{'tansig','tansig'}); %������ �����

    net.trainParam.epochs=1e5;
    net.trainParam.goal=1e-5;
    net.trainParam.lr=0.05;
    net.trainParam.show=10;
%    net.trainParam.showwindow=false;
    if nargin==3
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

    net=train(net,P,T);

    RES={net,ps,ts};
end
