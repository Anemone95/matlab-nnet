function [bestfitness,x]=EzGA(bound,OptFun,sizepop,attach)
    %% ��ʼ���Ŵ��㷨����
    %��ʼ������
    maxgen=50;                         %��������������������
    if nargin<3
        sizepop=500;
    end
    pcross=0.7;                       %�������ѡ��0��1֮��
    pmutation=0.2;                    %�������ѡ��0��1֮��
    
    [ra co]=size(bound);

    lenchrom=ones(1,ra);          %ÿ���������ִ����ȣ�����Ǹ���������򳤶ȶ�Ϊ1


    individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ��
    avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��
    bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
    bestchrom=[];                       %��Ӧ����õ�Ⱦɫ��

    %% ��ʼ����Ⱥ������Ӧ��ֵ
    % ��ʼ����Ⱥ
    for i=1:sizepop
        %�������һ����Ⱥ
        individuals.chrom(i,:)=Code(lenchrom,bound);   
        x=individuals.chrom(i,:);
        %������Ӧ��
        if nargin==4
            individuals.fitness(i)=OptFun(x,attach);   %Ⱦɫ�����Ӧ��
        else
            individuals.fitness(i)=OptFun(x);   %Ⱦɫ�����Ӧ��
        end
    end
    %����õ�Ⱦɫ��
    [bestfitness bestindex]=min(individuals.fitness);
    bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ��
    avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��
    % ��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
    traceA=[avgfitness bestfitness]; 
%y=x(1)^2-10*cos(2*pi*x(1))+10+x(2)^2-10*cos(2*pi*x(2))+10;

    %% ����Ѱ��
    % ������ʼ
    h=waitbar(0,'Evolving....');
    for i=1:maxgen
        % ѡ��
        individuals=Select(individuals,sizepop); 
        avgfitness=sum(individuals.fitness)/sizepop;
        %����
        individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
        % ����
        individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,[i maxgen],bound);
        
        % ������Ӧ�� 
        for j=1:sizepop
            x=individuals.chrom(j,:); %����
            if nargin==4
                individuals.fitness(j)=OptFun(x,attach);   %Ⱦɫ�����Ӧ��
            else
                individuals.fitness(j)=OptFun(x);   %Ⱦɫ�����Ӧ��
            end
        end
        
      %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
        [newbestfitness,newbestindex]=min(individuals.fitness);
        [worestfitness,worestindex]=max(individuals.fitness);
        % ������һ�ν�������õ�Ⱦɫ��
        if bestfitness>newbestfitness
            bestfitness=newbestfitness;
            bestchrom=individuals.chrom(newbestindex,:);
        end
        individuals.chrom(worestindex,:)=bestchrom;
        individuals.fitness(worestindex)=bestfitness;
        
        avgfitness=sum(individuals.fitness)/sizepop;
        
        traceA=[traceA;avgfitness bestfitness]; %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
        waitbar(i/maxgen,h,sprintf('Now Generation:%d',i));
        if i>30
            if sum(diff(traceA(end-30:end,2)))==0
                break
            end
        end
    end
    close(h)
    %��������

    %% �������
    [r c]=size(traceA);
    plot([1:r]',traceA(:,2),'r-');
    title('Fitness curve','fontsize',12);
    xlabel('Evolutionary generation','fontsize',12);ylabel('Fitness','fontsize',12);
    axis([0,maxgen,0,1])
    x=bestchrom;
    % ������ʾ
end
