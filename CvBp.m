function perfectNet=CvBp(P,T)
    pdata=P';
    tdata=T';

    pTrain=P;
    tTrain=T;

    n=10;
    %% ½»²æÑéÖ¤
    mse_max=10e30;
    desiredInput=[];
    desiredOutput=[];

    num=5;
    indices = crossvalind('Kfold',length(pTrain),num);
    for i = 1:num
        perfp=[];
        disp(['The result of ',num2str(i),'/',num2str(num)])
        test = (indices == i); trainA = ~test;
        pCvTrain=pTrain(trainA,:);
        tCvTrain=tTrain(trainA,:);
        pCvTest=pTrain(test,:);
        tCvTest=tTrain(test,:);
        pCvTrain=pCvTrain';
        tCvTrain=tCvTrain';
        pCvTest= pCvTest';
        tCvTest= tCvTest';

        nett=feedforwardnet(n);
        % net.trainParam.epochs=100000;
        % net.trainParam.show=200;
        % net.trainParam.goal=1e-4;
        % net=train(net,desired_input,desired_output);
        nett=train(nett,pCvTrain,tCvTrain);
        testOut=nett(pCvTest);
        perf=perform(nett,testOut,tCvTest);
        if mse_max>perf
            perfectNet=nett;
            mse_max=perf;
            desiredInput=pCvTrain;
            desiredOutput=tCvTrain;
        end
    end
end
