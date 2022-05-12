clear global ; clear; close all;

f = @(x) x.^2;

y = [2 3 4 5 6 7]';

x = [0.9636    0.6895    0.0798    0.6942    0.8498    0.6080]';
v = [1 1 1 1 1 1]'/64;

% Or try:
%x = rand(6,1);
%v = [1 1 1 1 1 1]'/128;


op = AO('options');
op.fun = f;
op.x0  = x;
op.V   = v;
op.y   = y;
op.step_method = 1;
op.criterion = -2.7;
op.maxit = 56;
op.hypertune=1;
op.objective='rmse';

[X,F,CV] = AO(op);

[y(:) f(X)]

bub = rescale(abs(100*(y-f(X))),20,50);
figure('position',[1078 283 482 695]);
subplot(211);scatter(y,f(X),bub,'filled');lsline;
xlim([1 8]);ylim([1 8]);xlabel('truth');ylabel('optimiser estimate');
subplot(212); b=bar([y f(X)]);b(1).FaceColor=[.4 .4 .4];
b(2).FaceColor=[1 .4 .4];legend({'truth' 'optim'});