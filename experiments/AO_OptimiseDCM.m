
% Use AO.m to optimise a fully-specified (spectral) DCM...
%----------------------------------------------------------

% The model function
fun = @(p) spm_vec( feval(DCM.M.IS,spm_unvec(p,DCM.M.pE),DCM.M,DCM.xU) );

% Parameters and variances
p = spm_vec(DCM.M.pE);
c = spm_vec(DCM.M.pC);

% Get and fill in AO options struct
opts     = AO('options');
opts.fun = fun;
opts.x0  = p(:);
opts.V   = c;
opts.y   = DCM.xY.y{:};

opts.maxit       = 36;
opts.inner_loop  = 12*4;
opts.Q           = [];
opts.criterion   = -500;
opts.min_df      = 1e-12;
opts.order       = 2;
opts.writelog    = 0;
opts.objective   = 'fe';
opts.ba          = 0;
opts.im          = 1;
opts.step_method = 1;

% Run the optimiser...
%[X,F,CP,Pp,History] = AO(opts);

% or fit by NLLS regression
%-------------------------------------
options = statset;
options.Display = 'iter';
options.TolFun  = 1e-6;
options.MaxIter = 128;
options.FunValCheck = 'off';
options.DerivStep = 1e-12;

funfun = @(b,p) fun(b.*p);

b = ones(size(opts.x0));

% fit using nonlinear least squares regression
[BETA,R,J,COVB,MSE] = atcm.optim.nlinfit(full(opts.x0),...
    full(opts.y),funfun,b,options);
