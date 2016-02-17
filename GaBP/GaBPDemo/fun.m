function perf = fun(x,attach)
%函数用于计算粒子适应度值
%x           input           输入粒子 
%y           output          粒子适应度值 
    [net,ps,ts,perf]=EzBP(attach{1},attach{2},x',attach{3},attach{4});
