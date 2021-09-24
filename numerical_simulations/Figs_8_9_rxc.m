clear all;
clc;
close all;


t = 0:0.05:1.05;       % time


% Decoidng probs are determined by EW_3classes_used_for_Fig7.m and NOW_3classes_used_for_Fig7.m - whose
% results are following mat files

load('NOW_probs.mat')
load('EW_probs.mat')


%ws_title = 'ws/rxc_3classes_extended_scaled_9packets_40_35_25.mat';
rxc_all_sim(Pd1_NOW,Pd2_NOW,Pd3_NOW, Pd1_EW,Pd2_EW,Pd3_EW,N, t);





function [] = rxc_all_sim(Pd1_NOW,Pd2_NOW,Pd3_NOW, Pd1_EW,Pd2_EW,Pd3_EW,N, t)
s = 300;
m = 900;

% N = 30; % number of workers
lambda = 1; % parameter for exponential latency model

mnx = [0 0 0];      % mean and variances for each row and column type (H,M,L)
varx = [10 1 0.1];

% processing of earlier obtained  decoding probabilities....
Pd = [Pd1_EW; Pd1_EW; Pd2_EW; Pd2_EW; Pd3_EW; Pd3_EW];
Pd = Pd(:,1:N);
Pd_EW = [zeros(6,1) Pd];

% load('ws_NOW_3classes_2_2_2.mat')

Pd = [Pd1_NOW; Pd1_NOW; Pd2_NOW; Pd2_NOW; Pd3_NOW; Pd3_NOW];
Pd = Pd(:,1:N);
Pd_NOW = [zeros(6,1) Pd];

% mean and variance table for each class
tab_class = zeros(6,4);
tab_class(1,:) = [mnx(1) varx(1) mnx(1) varx(1)];
tab_class(2,:) = [mnx(1) varx(1) mnx(2) varx(2)];
tab_class(3,:) = [mnx(2) varx(2) mnx(2) varx(2)];
tab_class(4,:) = [mnx(1) varx(1) mnx(3) varx(3)];
tab_class(5,:) = [mnx(2) varx(2) mnx(3) varx(3)];
tab_class(6,:) = [mnx(3) varx(3) mnx(3) varx(3)];

nc = [1 2 1 2 2 1]; % number of packets in each class

energy = allExp(nc,tab_class,lambda,s,m);   % E[C] = E[AB]



% MSE calculations for EW
MSE_EW_exp_rxc = zeros(1,length(t));
for tx = 1:length(t)
    for x = 0:N
        MSE_EW_exp_rxc(1,tx) = MSE_EW_exp_rxc(1,tx) + probRx(N,x,t(tx),lambda)*condMSE(Pd_EW,x,nc,tab_class,lambda,s,m);
    end
end
normMSE_EW_exp_rxc = (MSE_EW_exp_rxc)/energy;    %normalization

% MSE calculations for NOW
MSE_NOW_exp_rxc = zeros(1,length(t));
for tx = 1:length(t)
    for x = 0:N
        MSE_NOW_exp_rxc(1,tx) = MSE_NOW_exp_rxc(1,tx) + probRx(N,x,t(tx),lambda)*condMSE(Pd_NOW,x,nc,tab_class,lambda,s,m);
    end
end
normMSE_NOW_exp_rxc = (MSE_NOW_exp_rxc)/energy; %normalization

MDS_prob_rxc = zeros(1,length(t));
for tx = 1:length(t)
    for x = 0:sum(nc)-1
        MDS_prob_rxc(1,tx) = MDS_prob_rxc(1,tx) + probRx(N,x,t(tx),lambda);%*condMSE(Pd_NOW,x,nc,tab_class,lambda,s,m);
    end
end


% plot of MSEs
figure;
semilogy(t, normMSE_NOW_exp_rxc, '-*', 'linewidth',1.5);
hold on;
semilogy(t, normMSE_EW_exp_rxc, '--o','color',[0.9290, 0.6940, 0.1250] , 'linewidth',1.5);
hold on;
semilogy(t, MDS_prob_rxc, '-.dk', 'linewidth',1.5);



xlabel('Time (t)');
ylabel('Normalized MSE(t)');
title('UEP with three classes (r \times c)');
% title({'UEP with three classes, (H,M,L) = (2,2,2) for both A and B','Packets in each class: (12,12,12)', 'Window selection ditr: \Gamma = [0.45 0.35 0.2]', 'Exponential latency model with \lambda= 1'})
legend('NOW-UEP', 'EW-UEP', 'MDS');
grid on;
hold on
x0=10;
y0=10;
width=650;
height=250
set(gcf,'position',[x0,y0,width,height])
% xlim([0 0.025]);

MSE_N_EW_rxc = ones(1,length(N));
MSE_N_NOW_rxc = ones(1,length(N));


for ii = 1:N
    
    MSE_N_EW_rxc(1,ii+1) = condMSE(Pd_EW,ii,nc,tab_class,lambda,s,m)/energy;
    MSE_N_NOW_rxc(1,ii+1) = condMSE(Pd_NOW,ii,nc,tab_class,lambda,s,m)/energy;
    
end


MDS_mse_rxc = zeros(1,N+1);
MDS_mse_rxc(1:sum(nc)) = 1;



figure;
semilogy(0:N, MSE_N_NOW_rxc, '-*', 'linewidth',1.5);
hold on;
semilogy(0:N, MSE_N_EW_rxc, '--o','color',[0.9290, 0.6940, 0.1250] , 'linewidth',1.5);

xlabel('Number of received packets');
ylabel('Normalized MSE(N_r)');
title('UEP with three classes (r \times c)');
% title({'UEP with three classes, (H,M,L) = (2,2,2) for both A and B','Packets in each class: (12,12,12)', 'Window selection ditr: \Gamma = [0.45 0.35 0.2]', 'Exponential latency model with \lambda= 1'})
legend('NOW-UEP', 'EW-UEP');
grid on;
hold on
x0=10;
y0=10;
width=650;
height=250
set(gcf,'position',[x0,y0,width,height])



% save(ws_title);
end


function out = exp_cdf(lambda,t)
if t>=0
    out = 1-exp(-lambda*t);
else
    out = 0;
end

end

% Probability of receiving x packets at time t
function out = probRx(N,x,t,lambda)
F = exp_cdf(lambda,t);
out = (nchoosek(N,x))*((1-F)^(N-x))*(F^(x));
end


function out = probRx_poisson(N,x,t,lambda)
out = ((lambda*t)^x)*exp(-lambda*t)/factorial(x);
end

% helping function for MSE calculation - also defined in the paper
function out = f(s,m,mnA, varA, mnB, varB)
out = (s^2)*m*(varA+mnA^2)*(varB+mnB^2) + (s^2)*m*(m-1)*(mnA^2)*(mnB^2);
%*(varA*varB+varA*(mnB^2)+(mnA^2)*varB+m*(mnA^2)*(mnB^2));
end


% Conditional MSE given Nr packets are received: E[||C-\hatC||_F^2 | Nr]
function out = condMSE(Pd,Nr,nc,tab_class,lambda,s,m)
out = 0;
for c = 1:6
    out = out + (nc(c))*(1-Pd(c,Nr+1))*f(s,m,tab_class(c,1), tab_class(c,2), tab_class(c,3), tab_class(c,4));
end
end


% This is just for E[||C||_F^2]
function out = allExp(nc,tab_class,lambda,s,m)
out = 0;
for c = 1:6
    out = out + (nc(c))*f(s,m,tab_class(c,1), tab_class(c,2), tab_class(c,3), tab_class(c,4));
end
end


