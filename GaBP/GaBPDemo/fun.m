function perf = fun(x,attach)
%�������ڼ���������Ӧ��ֵ
%x           input           �������� 
%y           output          ������Ӧ��ֵ 
    [net,ps,ts,perf]=EzBP(attach{1},attach{2},x',attach{3},attach{4});
