function y = fun(x,attach)
%�������ڼ���������Ӧ��ֵ
%x           input           �������� 
%y           output          ������Ӧ��ֵ 


bpRes=EzBP(attach{1},attach{2},x',attach{3},attach{4});
y=bpRes{3};
