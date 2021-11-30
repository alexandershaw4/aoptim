clear global ; clear; close all;

% Example function with interacting states (params)...
%-------------------------------------------------------------
f  = @(x) ( sum(toeplitz(x)) )' ;
f  = @(x) (pinv(toeplitz(x)*toeplitz(x)')*toeplitz(x)) ;

x  = 2:12; % actual optimsed values to find
y  = f(x); % data given correct parameters

rng default;

x0 = randn(size(x));     % random start positions
V  = ones(size(x0))/512; % variances/step sizes

% Setting up the optimiser
%-------------------------------------------------------------
op = AO('options');  % this returns the optimiser input options structure
op.step_method = 0;  % aggressive steps = 1, careful = 3, vanilla = 4.
op.fun = f;          % function/model
op.x0  = x0(:);      % start values
op.y   = y;%(:);       % data we're fitting (for computation of objective fun)
op.V   = V(:);       % corresponding vars/step sizes for each param (x0)

op.step_method = 1;   % aggressive steps = 1, careful = 3, vanilla = 4.
op.maxit       = 128; % maximum number of iterations

op.inner_loop = 2;
op.BTLineSearch = 0;
op.fsd=0;
op.DoMLE = 0;
op.doimagesc=1;

[X,F,CV] = AO(op);    % RUN IT

% Compare data with prediction
[f(x) f(X)]