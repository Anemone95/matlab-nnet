function flag=test(lenchrom,bound,code)
% lenchrom   input : Ⱦɫ�峤��
% bound      input : ������ȡֵ��Χ
% code       output: Ⱦɫ��ı���ֵ

x=code; %�Ƚ���
flag=1;
ra=size(bound,1);
for i=1:ra
    % if (x(1)<bound(1,1))&&(x(2)<bound(2,1))&&(x(1)>bound(1,2))&&(x(2)>bound(2,2))
        % flag=0;
    % end
    if x(i)<bound(i,1)||x(i)>bound(i,2)
        flag=0;
    end
end
     
