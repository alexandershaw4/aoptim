function [Q,X,F,CV] = fit_spectrum(w,y)

%-------------------------------------------------------------
y = real(y);

% make vector valued function and start points
fq = linspace(w(1),w(end),6);
P.Freq = round(fq(2:5));
P.Amp  = y(round(P.Freq))*0.75;
P.Wid  = [1 1 1 1]*2;
P.aper = [mean(y)/16*exp(1) 1];
x0 = (spm_vec(P));

V  = ~~spm_vec(P)/16;
V(1:4)=1/4;
V(9:12)=0;

f  = @(x) (makef(w,spm_unvec(abs(x),P)));

% Setting up the optimiser
%-------------------------------------------------------------
op = AO('options');  % this returns the optimiser input options structure
op.step_method = 1;  % aggressive steps = 1, careful = 3, vanilla = 4.
op.fun = f;          % function/model
op.x0  = x0(:);      % start values
op.y   = y(:);       % data we're fitting (for computation of objective fun)
op.V   = V(:);       % corresponding vars/step sizes for each param (x0)

op.maxit        = 16; % maximum number of iterations
op.inner_loop   = 30;
op.BTLineSearch = 0;
op.DoMLE        = 0;
op.ismimo=0;
op.hyperparams=1;
op.im=1;
op.fsd=0;
%op.FS = @(x) x(:).^2.*(1:length(x))';
op.criterion = -inf;
op.doparallel=0;

%op.fsd=-1;
op.corrweight=1;

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
P
Q = spm_unvec(X,P)



