clear global ; clear; close all;

% Example function: fit a 4-dim Gaussian Mixture Model by minimising the
% free energy
%
% AS2021


%-------------------------------------------------------------
w      = 2:90;
S.Freq = [8  14 50 60]; % true values x
S.Amp  = [10 8  5  5 ]; % true values y
S.Wid  = [1  3  2  2];  % true values width

Y = makef(w,S); % truth...

% make vector valued function and start points
P      = S;
P.Freq = [4 12 40 66];
P.Amp  = [2  2  2  2];
P.Wid  = [1 1 1 1];
x0 = (spm_vec(P));

V  = ~~spm_vec(P)/16;
V(1:4)=1/4;
V(9:12)=0;
f  = @(x) (makef(w,spm_unvec(abs(x),S)));

V(3:4)=1/16;

% Setting up the optimiser
%-------------------------------------------------------------
op = AO('options');  % this returns the optimiser input options structure
op.step_method = 1;  % aggressive steps = 1, careful = 3, vanilla = 4.
op.fun = f;          % function/model
op.x0  = x0(:);      % start values
op.y   = Y(:);       % data we're fitting (for computation of objective fun)
op.V   = V(:);       % corresponding vars/step sizes for each param (x0)

op.maxit        = 128; % maximum number of iterations
op.inner_loop   = 30;
op.BTLineSearch = 0;
op.DoMLE        = 0;
op.ismimo=1;
op.hyperparams=1;
op.im=1;
op.fsd=0;
op.FS = @(x) x(:).^2.*(1:length(x))';
op.criterion = -inf;
op.doparallel=0;

% Step 1. Optimise the x- and y- values of the GMM but holding the width
% constant...
%--------------------------------------------------------------------
[X,F,CV,~,Hi] = AO(op);    % RUN IT

% Step 2. Optimise the x- and y- values and the width
%--------------------------------------------------------------------
op.x0 = X;
op.V(9:12)=1/8;       % enable width to vary & continue optimisation
[X,F,CV] = AO(op);    % RUN IT

% Compare data with prediction
S
spm_unvec(X,S)



