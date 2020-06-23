clear global ; clear; close all;

% Example function with interacting states...
f  = @(x) ( sum(toeplitz(x)) )' ;

x  = 2:12; % actual optimsed values to find
y  = f(x); % data given correct parameters

rng default;

x0 = randn(size(x));     % random start positions
V  = ones(size(x0))/512; % variances/step sizes

op = AO('options');
op.step_method = 3;
op.fun = f;
op.x0  = x0(:);
op.y   = y(:);
op.V   = V(:);

op.step_method = 3;
op.maxit       = 128;

[X,F,CV] = AO(op);

% Compare data with prediction
[f(x) f(X)]