clear all;
clc;
close all;


delta = [0.4, 0.35, 0.25];  % selection probability of the indows (in order)
k = [3 3 3];  % number of packets in each window (k(1): 1st class, k(2): second class)

nums = [1:30];


Pd1_EW = zeros(1,length(nums));
Pd2_EW = zeros(1,length(nums));
Pd3_EW = zeros(1,length(nums));




count = 1;
for N = nums
    
    for i1 = 0:N
        for i2  = 0:N-i1
       
            n = [i1 i2  N-(i1+i2)];
            R = Rln(n,k);
            bol = EW_bol(k,R);
            Pd1_EW(1,count) = Pd1_EW(1,count)+ (Pdelta(n, delta)*(bol(1)));
            Pd2_EW(1,count) = Pd2_EW(1,count)+ (Pdelta(n, delta)*(bol(2)));
            Pd3_EW(1,count) = Pd3_EW(1,count)+ (Pdelta(n, delta)*(bol(3)));
            
            
        end
  
    end
    a = 0;
    count = count+1;
    disp(['EW, N: ', int2str(N)]);
end


save('EW_probs');
figure;
plot(nums, Pd1_EW,'-.o',nums, Pd2_EW,'-.x',nums, Pd3_EW,'-d','linewidth',1.5,'MarkerIndices', 1:2:length(nums));
legend('P_{d,1}(N) - EW', 'P_{d,2}(N) - EW', 'P_{d,3}(N) - EW','location','best')
xlabel('received packets (N)');
ylabel('P_{d,c}(N)s');
ylim([0, 1.02]);
title('Decoding probabilities');
grid on;

x0=10;
y0=10;
width=650;
height=400
set(gcf,'position',[x0,y0,width,height])



function [out] = Pdelta(n,delta)
out = factorial(sum(n));

for i = 1:length(n)
    out = out*(delta(i)^n(i))/factorial(n(i));
    if isnan(out)
        dd = 5;
    end
end

end


function [R] = Rln(n,K)
R1 = min([n(1),K(1)]);
R2 = min([R1+n(2),sum(K(1:2))]);
R3 = min([R2+n(3),sum(K(1:3))]);

R = [R1 R2 R3 ];
end

function [bol] = EW_bol(k,R)
bol1 = (R(1) == k(1));% | ((R(1)< k(1)) & (R(2) == sum(k(1:2)))) | ((R(1)< k(1)) & (R(3) == sum(k(1:3))));
for ii = 2:3
    xx = true;
    for jj = 1:ii-1
        xx = xx & (R(jj) < sum(k(1:jj)));
    end
    xx = xx & (R(ii) == sum(k(1:ii)));
    bol1 = bol1 | xx;
end



bol2 = (R(2) == sum(k(1:2)));
for ii = 3:3
    xx = true;
    for jj = 2:ii-1
        xx = xx & (R(jj) < sum(k(1:jj)));
    end
    xx = xx & (R(ii) == sum(k(1:ii)));
    bol2 = bol2 | xx;
end

bol3 = (R(3) == sum(k(1:3)));
bol = [bol1 bol2 bol3 ];

end


