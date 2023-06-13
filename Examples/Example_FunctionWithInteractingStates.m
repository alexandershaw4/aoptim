clear global ; clear; close all;

% Example function with interacting states (params)...
%-------------------------------------------------------------
f  = @(x) ( sum(toeplitz(x)) )' ;
f  = @(x) (pinv(toeplitz(x)*toeplitz(x)')*toeplitz(x)) ;

x  = 2:12; % actual optimsed values to find
y  = f(x); % data given correct parameters

rng default;

x0 = randn(size(x));     % random start positions
V  = ones(size(x0))/8; % variances/step sizes

% Setting up the optimiser
%-------------------------------------------------------------
op = AO('options');  % this returns the optimiser input options structure
op.fun = f;          % function/model
op.x0  = x0(:);      % start values
op.y   = y;%(:);       % data we're fitting (for computation of objective fun)
op.V   = V(:);       % corresponding vars/step sizes for each param (x0)

op.step_method = 1;   % aggressive steps = 1, careful = 3, vanilla = 4.
op.maxit       = 128; % maximum number of iterations

op.inner_loop = 2;
op.fsd=1;
op.doimagesc=1;

op.objective='gaussq';

op.hypertune=0;
op.rungekutta=8;
op.memory_optimise=0;
op.hyperparams=1;
op.isGaussNewtonReg = 0;

op.ismimo=1;
op.criterion = -inf;
op.gradtol = -1;

% set Q
w  = x;
X0 = spm_dctmtx(length(w),8);
Q  = speye(length(w)) - X0*X0';
Q = Q .* atcm.fun.AGenQn(f(x),8);
Q = abs(Q) + AGenQn(diag(Q),8);
op.Q = atcm.fun.AGenQn((Q(:)));

[X,F,CV] = AO(op);    % RUN IT

% Compare data with prediction
[f(x) f(X)]