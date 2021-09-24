clear all;
clc;
% close all;


delta = [0.40, 0.35, 0.25];  % selection probability of the indows (in order)
k = [3 3 3];  % number of packets in each window (k(1): 1st class, k(2): second class)

nums = [1:30];

Pd1_NOW = zeros(1,length(nums));
Pd2_NOW = zeros(1,length(nums));
Pd3_NOW = zeros(1,length(nums));



count = 1;
for N = nums

    for i1 = 0:N
        for i2  = 0:N-i1
            
            
            n = [i1 i2 N-(i1+i2)];
            
            Pd1_NOW(1,count) = Pd1_NOW(1,count)+ (Pdelta(n, delta)*(n(1) >= k(1)));
            Pd2_NOW(1,count) = Pd2_NOW(1,count)+ (Pdelta(n, delta)*(n(2)>= k(2)));
            Pd3_NOW(1,count) = Pd3_NOW(1,count)+ (Pdelta(n, delta)*(n(3) >= k(3)));
            
        end
        
        
        
    end
    
    count = count+1;
    disp(['NOW, N: ', int2str(N)]);
end


figure;
plot(nums, Pd1_NOW,'-.o',nums, Pd2_NOW,'-.x',nums, Pd3_NOW,'-d','linewidth',1.5,'MarkerIndices', 1:2:length(nums));
legend('P_{d,1}(N) - NOW', 'P_{d,2}(N) - NOW', 'P_{d,3}(N) - NOW','location','best')
xlabel('received packets (N)');
ylabel('P_{d,c}(N)s');
% xlim([0 102]);
ylim([0, 1.02]);
title('Decoding probabilities');
grid on;

x0=10;
y0=10;
width=650;
height=400
set(gcf,'position',[x0,y0,width,height])

save('NOW_probs')

function [out] = Pdelta(n,delta)
out = factorial(sum(n));
for i = 1:length(n)
    out = out*(delta(i)^n(i))/factorial(n(i));
end
end





