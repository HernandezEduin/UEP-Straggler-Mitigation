clear all;
clc;
close all;

MC_iter = 100;
t = 0:0.075:1.05;       % time

% Decoidng probs are determined by EW_3classes_used_for_Fig7.m and NOW_3classes_used_for_Fig7.m - whose
% results are following mat files

load('NOW_probs.mat')
load('EW_probs.mat')

cxr_all_sim(Pd1_NOW,Pd2_NOW,Pd3_NOW, Pd1_EW,Pd2_EW,Pd3_EW,N,MC_iter,t)






function [] = cxr_all_sim(Pd1_NOW,Pd2_NOW,Pd3_NOW, Pd1_EW,Pd2_EW,Pd3_EW,N, MC_iter,t)


% MC_iter = 3;       % Monte carlo iter for exact calculation

lambda = 1; % parameter for exponential latency model

mnx = [0 0 0 ];      % mean and variances for each row and column type (H,M,L)
varx = [10 1 0.1];


% A: U x MH
% B: MH x Q
U = 900;
M = 9;
H = 100;
Q = U;

energy = energyCal(U,H,Q,varx,mnx,MC_iter);      % calculate the E[C^2] for normalization later

% processing of earlier obtained  decoding probabilities....
Pd = [Pd1_EW; Pd1_EW; Pd1_EW; Pd2_EW; Pd2_EW; Pd2_EW; Pd3_EW; Pd3_EW; Pd3_EW];
Pd = Pd(:,1:N);
Pd_EW = [zeros(9,1) Pd];


Pd = [Pd1_NOW; Pd1_NOW; Pd1_NOW; Pd2_NOW ; Pd2_NOW; Pd2_NOW; Pd3_NOW; Pd3_NOW; Pd3_NOW];
Pd = Pd(:,1:N);
Pd_NOW = [zeros(9,1) Pd];

% mean and variance table for each class
tab_class = zeros(9,4);
tab_class(1,:) = [mnx(1) varx(1) mnx(1) varx(1)];
tab_class(2,:) = [mnx(1) varx(1) mnx(1) varx(1)];
tab_class(3,:) = [mnx(1) varx(1) mnx(1) varx(1)];
tab_class(4,:) = [mnx(2) varx(2) mnx(2) varx(2)];
tab_class(5,:) = [mnx(2) varx(2) mnx(2) varx(2)];
tab_class(6,:) = [mnx(2) varx(2) mnx(2) varx(2)];
tab_class(7,:) = [mnx(3) varx(3) mnx(3) varx(3)];
tab_class(8,:) = [mnx(3) varx(3) mnx(3) varx(3)];
tab_class(9,:) = [mnx(3) varx(3) mnx(3) varx(3)];


nc = [1 1 1 1 1 1 1 1 1 ]; % number of packets in each class



% MSE calculations for EW
MSE_EW_exp_cxr = zeros(1,length(t));
MSE_EW_exact_cxr = zeros(1,length(t));
for tx = 1:length(t)
    tic
    for x = 0:N
        
        MSE_EW_exp_cxr(1,tx) = MSE_EW_exp_cxr(1,tx) + probRx(N,x,t(tx),lambda)*condMSE(Pd_EW,x,nc,tab_class,lambda,U*M*H*Q);
        
        MSE_EW_exact_cxr(1,tx) = MSE_EW_exact_cxr(1,tx) +  probRx(N,x,t(tx),lambda)*monte_carlo_new(Pd_EW,x,M,U,H,Q,tab_class,MC_iter);
        
    end
    toc
    time_tc = tx
end
normMSE_EW_exp_cxr = (MSE_EW_exp_cxr)/energy;    %normalization
normMSE_EW_exact_cxr = (MSE_EW_exact_cxr);%/energy;    %normalization

% MSE calculations for NOW
MSE_NOW_exp_cxr = zeros(1,length(t));
MSE_NOW_exact_cxr = zeros(1,length(t));
for tx = 1:length(t)
    tic
    for x = 0:N
        MSE_NOW_exp_cxr(1,tx) = MSE_NOW_exp_cxr(1,tx) + probRx(N,x,t(tx),lambda)*condMSE(Pd_NOW,x,nc,tab_class,lambda,U*M*H*Q);
        MSE_NOW_exact_cxr(1,tx) = MSE_NOW_exact_cxr(1,tx) +  probRx(N,x,t(tx),lambda)*monte_carlo_new(Pd_NOW,x,M,U,H,Q,tab_class,MC_iter);
        
    end
    toc
    time_tc = tx
end
normMSE_NOW_exp_cxr = (MSE_NOW_exp_cxr)/energy; %normalization
normMSE_NOW_exact_cxr = (MSE_NOW_exact_cxr);%/energy; %normalization

MDS_prob_cxr = zeros(1,length(t));
for tx = 1:length(t)
    for x = 0:sum(nc)-1
        MDS_prob_cxr(1,tx) = MDS_prob_cxr(1,tx) + probRx(N,x,t(tx),lambda);%*condMSE(Pd_NOW,x,nc,tab_class,lambda,s,m);
    end
end




figure;
semilogy(t, normMSE_NOW_exact_cxr, '-*',t, normMSE_NOW_exp_cxr, '-x',t, normMSE_EW_exact_cxr, '--o',t, normMSE_EW_exp_cxr, '--s',t, MDS_prob_cxr, '-.dk', 'linewidth',1.5,'MarkerIndices', 1:5:length(t));
xlabel('Time (t)');
ylabel('Normalized MSE(t)');
title('UEP with three classes (c \times r)');
% title({'UEP with three classes, (H,M,L) = (2,2,2) for both A and B','Packets in each class: (12,12,12)', 'Window selection ditr: \Gamma = [0.45 0.35 0.2]', 'Exponential latency model with \lambda= 1'})
legend('NOW-UEP exact', 'NOW-UEP bound','EW-UEP exact', 'EW-UEP bound', 'MDS');
grid on;
hold on
x0=10;
y0=10;
width=650;
height=250;
set(gcf,'position',[x0,y0,width,height])
ylim([1e-5 1 ]);


MSE_N_EW_cxr = ones(1,length(N));
MSE_N_NOW_cxr = ones(1,length(N));
MSE_N_EW_exact_cxr = ones(1,length(N));
MSE_N_NOW_exact_cxr = ones(1,length(N));

for ii = 0:N
    
%     MSE_N_EW_cxr(1,ii+1) = condMSE(Pd_EW,ii,nc,tab_class,lambda,U*M*H*Q);
%     MSE_N_NOW_cxr(1,ii+1) = condMSE(Pd_NOW,ii,nc,tab_class,lambda,U*M*H*Q);
    MSE_N_NOW_exact_cxr(1,ii+1) = monte_carlo_new(Pd_NOW,ii,M,U,H,Q,tab_class,MC_iter);
    MSE_N_EW_exact_cxr(1,ii+1) = monte_carlo_new(Pd_EW,ii,M,U,H,Q,tab_class,MC_iter);
    
end



figure;
semilogy(0:N, MSE_N_NOW_exact_cxr, '-*', 'linewidth',1.5);
hold on;
semilogy(0:N, MSE_N_EW_exact_cxr, '--o','color',[0.9290, 0.6940, 0.1250] , 'linewidth',1.5);

xlabel('Number of received packets (N_r)');
ylabel('Normalized MSE(N_r)');
title('UEP with three classes (c \times r)');
% title({'UEP with three classes, (H,M,L) = (2,2,2) for both A and B','Packets in each class: (12,12,12)', 'Window selection ditr: \Gamma = [0.45 0.35 0.2]', 'Exponential latency model with \lambda= 1'})
legend('NOW-UEP', 'EW-UEP');
grid on;
hold on
x0=10;
y0=10;
width=650;
height=250
set(gcf,'position',[x0,y0,width,height])
xlim([0 15]);





MDS_mse_cxr = zeros(1,N+1);
MDS_mse_cxr(1:sum(nc)) = 1;



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
function out = f(varA,varB,mult_umhq)
out = varA*varB*mult_umhq;
%*(varA*varB+varA*(mnB^2)+(mnA^2)*varB+m*(mnA^2)*(mnB^2));
end


% Conditional MSE given Nr packets are received: E[||C-\hatC||_F^2 | Nr]
function out = condMSE(Pd,Nr,nc,tab_class,lambda,mult_umhq)
out = 0;
for c = 1:length(nc)
    out = out + (nc(c))*(1-Pd(c,Nr+1))*f(tab_class(c,2), tab_class(c,4),mult_umhq);
end
end

function out = monte_carlo_new(P,n,M,U,H,Q,tabx,iter)
out = 0;

for ii = 1:iter
    A = zeros(U,M*H);
    B = zeros(M*H,Q);
    Ad = zeros(U,M*H);
    Bd = zeros(M*H,Q);
    for jj = 1:length(tabx)
        pp = rand;
        Ajj = randn(U,H)*sqrt(tabx(jj,2));
        Bjj = randn(H,Q)*sqrt(tabx(jj,4));
        Ad(:,(jj-1)*H+1:jj*H) = Ajj;
        Bd((jj-1)*H+1:jj*H,:) = Bjj;
        %          P(jj,n+1)
        if pp > P(jj,n+1)
            A(:,(jj-1)*H+1:jj*H) = Ajj;
            B((jj-1)*H+1:jj*H,:) = Bjj;
        end
    end
    out = out + sum(sum((A*B).^2))/sum(sum((Ad*Bd).^2));
end
out = out/iter;

end

% for E[C^2]
function energy = energyCal(U,H,Q,varx,mnx,iter)
energy = 0;
iter = 10;
for ii = 1:iter
    A1 = randn(U,H)*sqrt(varx(1));
    A2 = randn(U,H)*sqrt(varx(1));
    A3 = randn(U,H)*sqrt(varx(1));
    A4 = randn(U,H)*sqrt(varx(2));
    A5 = randn(U,H)*sqrt(varx(2));
    A6 = randn(U,H)*sqrt(varx(2));
    A7 = randn(U,H)*sqrt(varx(3));
    A8 = randn(U,H)*sqrt(varx(3));
    A9 = randn(U,H)*sqrt(varx(3));
    A = [A1 A2 A2 A4 A5 A6 A7 A8 A9];
    
    B1 = randn(H,Q)*sqrt(varx(1));
    B2 = randn(H,Q)*sqrt(varx(1));
    B3 = randn(H,Q)*sqrt(varx(1));
    B4 = randn(H,Q)*sqrt(varx(2));
    B5 = randn(H,Q)*sqrt(varx(2));
    B6 = randn(H,Q)*sqrt(varx(2));
    B7 = randn(H,Q)*sqrt(varx(3));
    B8 = randn(H,Q)*sqrt(varx(3));
    B9 = randn(H,Q)*sqrt(varx(3));
    
    
    B = [B1; B2; B3; B4; B5; B6; B7; B8; B9];
    C = A*B;
    energy = energy + sum(sum(C.^2));
end
energy = energy/iter;

end
