function [X,F] = ExampleFit_GMM(S)
%clear global ; clear; close all;

% Example function: fit a 4-dim Gaussian Mixture Model by minimising the
% free energy
%
% AS2021

st = tic;

%-------------------------------------------------------------
w      = 2:90;

if nargin < 1
    S.Freq = [8  14 50 60]; % true values x
    S.Amp  = [10 8  5  5 ]; % true values y
    S.Wid  = [1  3  2  2];  % true values width
end

Y = makef(w,S); % truth...

% make vector valued function and start points
P      = S;
P.Freq = [4 12 40 66];
P.Amp  = [2  2  2  2];
P.Wid  = [1 1 1 1];
x0 = (spm_vec(P));

V  = ~~spm_vec(P)/8;
V(9:12)=0;
f  = @(x) (makef(w,spm_unvec(abs(x),S)));

% f  = @(x) (makef(w,spm_unvec(abs(x),S))).*hnoisebasis(length(w),x(13:14))';
% 
% x0(13:14)=1;
% V(13:14)=1/8;

V(3:4)=1/16;

%V = V*8;

% Setting up the optimiser
%-------------------------------------------------------------
op = AO('options');  % this returns the optimiser input options structure
op.step_method = 9;  % aggressive steps = 1, careful = 3, vanilla = 4.
op.fun = f;          % function/model
op.x0  = x0(:);      % start values
op.y   = Y(:);       % data we're fitting (for computation of objective fun)
op.V   = V(:);       % corresponding vars/step sizes for each param (x0)

op.maxit        = 16; % maximum number of iterations
op.inner_loop   = 10;
op.DoMLE        = 0; % do use MLE for param estimation
op.ismimo       = 1; % compute jacobian on full model output not objective
op.hyperparams  = 1; % estimate noise hyperparameters
op.im           = 1; % use momentum acceleration
op.fsd          = 0; % fixed-step for derivative computation
op.FS = @(x) x(:).^2.*(1:length(x))';
op.FS = @(x) sqrt(x); % feature selection function

op.ahyper=1;
%op.nocheck=1;

%op.FS = @(x) [sqrt(x(:)); std(diff(x))/abs(mean(diff(x)))];

op.criterion  = -inf; 1e-3;
op.doparallel = 0; % compute stuff using parfor
op.DoMLE=0;
op.factorise_gradients = 1; % factorise/normalise grads
op.normalise_gradients=0;
op.objective='gauss_trace';%'mvgkl';%'log_mvgkl';%'mvgkl'; % set objective fun: multivariate gaussian KL div
op.EnforcePriorProb=0;
op.order=1; % second order gradients
%

op.do_gpr=0; % dont do gaussian process regression to learn Jac
op.hypertune=1; % do hypertuning 
op.rungekutta=8; % do an RK-line search
%op.bayesoptls=6;
op.updateQ=1; % update the precision matrix on each iteration
op.Q = eye(length(w));

op.nocheck=0;

% % since I know y is a low-rank 1D GMM, I can set Q to have only 8 comps
% NC = 12;
% [~,~,QQ] = atcm.fun.approxlinfitgaussian(Y(:),[],[],NC);
%     for iq = 1:NC; QQ{iq} = QQ{iq}'*QQ{iq}; end
% op.Q = QQ;

% [~,I,Qc] = atcm.fun.approxlinfitgaussian(Y,[],[],2);
% Qc = cat(1,Qc{:});
% %Qc = VtoGauss(real(DCM.xY.y{:}));
% % fun = @(x) full(atcm.fun.HighResMeanFilt(diag(x),1,4));
% for iq = 1:size(Qc,1); 
%     %Qc(iq,:) = Qc(iq,:) ./ max(Qc(iq,:));
%     QQ{iq} = Qc(iq,:)'*Qc(iq,:); 
% end
% 
% op.Q = QQ;

%op.WeightByProbability=1;

% or generate a confounds Q matrix
X0 = spm_dctmtx(length(w),8);
Q  = speye(length(w)) - X0*X0';
Q = Q .* atcm.fun.AGenQn(f(spm_vec(S)),8);
Q = abs(Q) + AGenQn(diag(Q),8);
op.Q = Q;

op.memory_optimise=1; % remember & include (optimise) prev update steps when considering new steps
op.crit = [0 0 0 0];


% make regular saves of the optimimsation
op.save_constant = 0;

op.isNewton=1;
op.isGaussNewton=0;
op.isQuasiNewton=0;
op.isNewtonReg=0;
op.isTrust=0;
%op.forcenewton=1;


op.makevideo=0;

%op.predictionerrorupdate=1;

% use QR factorisation to predict dx from Jaco and residual
%op.isQR=1;

%op.NatGrad = 1;

%op.nocheck=1;

%op.Q = (eye(length(w)) + AGenQ(Y));

% Step 1. Optimise the x- and y- values of the GMM but holding the width
% constant...
%--------------------------------------------------------------------
[X,F,CV,~,Hi] = AO(op);    % RUN IT

% % Step 2. Optimise the x- and y- values and the width
% %--------------------------------------------------------------------
% op.x0 = X;
% op.V(9:12)=1/8;       % enable width to vary & continue optimisation
% [X,F,CV] = AO(op);    % RUN IT

% Compare data with prediction
S
spm_unvec(X,S)

toc(st)
