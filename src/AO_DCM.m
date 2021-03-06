function [EP,F,CP,Pp,History] = AO_DCM(P,DCM,niter,method,sm,hyper)
% A wrapper for fitting DCMs with AO.m curvature optimisation routine.
% 
% Input parameter structure, P and a fully specified DCM structure:
%
%    [EP,F,CP] = AO_DCM(P,DCM)
%
%    [EP,F,CP,Pp,History] = AO_DCM(P,DCM,niter,mimo,method)
%
%  Inputs:
%  P = prior parameter struct
%  DCM    = fully specified normal DCM struct
%  niter  = number of iterations 
%  mimo   = set to 0
%  method = objective fun: 'fe' or 'sse' (free energy or squared error)
%
%  Outputs:
%  EP = fitted parameters / posteriors
%  F  = fit value
%  CP = parameter covariance (laos see global aopt.J)
%  Pp = posterior probabilities
%  history = struct with optimisation steps history
%
% Dependencies: SPM
% AS

global DD

if nargin < 6 || isempty(hyper)
    hyper = 1;
end

if nargin < 5 || isempty(sm) % search_method option (1,2 or 3)
    sm = 3;
end
if nargin < 4 || isempty(method)
    method = 'fe';
end
if nargin < 3 || isempty(niter)
    niter = 128;
end

DD    = DCM;
DD.SP = P;
b0    = P;
P     = spm_vec(P);
V     = spm_vec(DCM.M.pC);
ip    = find(V);
cm    = zeros(length(V),length(ip));

% make and store a mapping matrix
for i = 1:length(ip)
    cm(ip(i),i) = 1;
end

% to pass to f(�x)
DD.P  = P;
DD.V  = V;
DD.cm = cm;

fprintf('Performing AO optimisation: ');

p = ones(length(ip),1);
c = V(ip);

switch lower(method)
    case 'sse'
    % use this to minimise SSE:
    fprintf('Minimising SSE\n');
    %[X,F,CP,Pp,History]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],1e-3,1e-12,2,0,'sse');
    
    % Get and fill in AO options struct
    opts     = AO('options');      
    opts.fun = @fakeDM; 
    opts.x0  = p(:);
    opts.V   = c;
    opts.y   = DCM.xY.y;
    
    opts.maxit       = niter;
    opts.inner_loop  = 12*4;
    opts.Q           = [];
    opts.criterion   = -inf;%-500;%-inf;
    opts.min_df      = 1e-12;
    opts.order       = 2;
    opts.writelog    = 0;
    opts.objective   = 'sse';
    opts.ba          = 0;
    opts.im          = 1;
    opts.step_method = sm;
    
    opts.force_ls=0;
    
    %opts.ismimo = 1;
    
    [X,F,CP,Pp,History] = AO(opts);       
    
    CP = atcm.fun.reembedreducedcovariancematrix(DCM,CP);
    
    case 'fe'
    % minimise free energy:
    fprintf('Minimising Free-Energy\n'); % 
    
    % Get and fill in AO options struct
    opts     = AO('options');      
    opts.fun = @fakeDM; 
    opts.x0  = p(:);
    opts.V   = c;
    opts.y   = DCM.xY.y;
    
    opts.maxit       = niter;
    opts.inner_loop  = 8;
    opts.Q           = [];
    opts.criterion   = -inf;%-500;%-inf;
    opts.min_df      = 1e-12;
    opts.order       = 2;
    opts.writelog    = 0;
    opts.objective   = 'fe';
    opts.ba          = 0;
    opts.im          = 1;
    opts.step_method = sm;
    
    opts.BTLineSearch=0;
    opts.force_ls=0;
    %opts.parallel=1;
    opts.hyperparams=hyper; % the third FE term - an ascent on the noise
    
    %[X,F,CP,Pp,History] = AO(opts);        
        
    [X,F,CP,Pp,History] = AO(opts);   
    
    CP = atcm.fun.reembedreducedcovariancematrix(DCM,CP);
    
    %[X,F,CP,Pp,History]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],-inf,1e-12,2,0,'fe',1,1,sm);
    %[X,F,CP,History]  = AOm(@fakeDM,p(:),c,DCM.xY.y);
    
    case 'logevidence'
    fprintf('Minimising -[log evidence]\n');
    %[X,F,CP,Pp,History]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],-inf,1e-12,2,0,'logevidence');
    
    % Get and fill in AO options struct
    opts     = AO('options');      
    opts.fun = @fakeDM; 
    opts.x0  = p(:);
    opts.V   = c;
    opts.y   = DCM.xY.y;
    
    opts.maxit       = niter;
    opts.inner_loop  = 12*4;
    opts.Q           = [];
    opts.criterion   = -inf;%-500;%-inf;
    opts.min_df      = 1e-12;
    opts.order       = 2;
    opts.writelog    = 0;
    opts.objective   = 'logevidence';
    opts.ba          = 0;
    opts.im          = 1;
    opts.step_method = sm;
    
    opts.force_ls=0;
    %opts.ismimo = 1;
    %opts.gradmemory = 1;
    
    %[X,F,CP,Pp,History] = AO(opts);        
        
    [X,F,CP,Pp,History] = AO(opts);       
    
    CP = atcm.fun.reembedreducedcovariancematrix(DCM,CP);
    
    case {'sample_fe' 'sampler_fe' 'fe_sampler' 'fe_sample' 'sample'}
    % sampling routine
    %[X,F]  = AOsample(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],-inf,1e-12,mimo,2,0,'fe');
    %CP=[];Pp=[];History=[]; 
    
        opts     = AOsample('options');      
        opts.fun = @fakeDM; 
        opts.x0  = p(:);
        opts.V   = c;
        opts.y   = DCM.xY.y;
        opts.criterion   = -inf;%-500;%-inf;
        [X,F] = AOsample(opts);   
        
    case 'control'
        
        mpcontrol(@fakeDM,p,c,DCM.xY.y);

end

% to ignore the variances/step sizes and allow AO to compute them:
%[X,F,CP,History]  = AO(@fakeDM,p(:),[],DCM.xY.y,niter,12*4,[],1e-3,1e-12,mimo,2);

% to use the flavour that uses a taylor-like expansion for parameter steps
% over iterations:
%[X,F,CP]  = AOt(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],1e-3,1e-12,0,4);

% to use normal AO.m but include an output precision operator (Q):
%[X,F,CP]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,DCM.xY.Q,1e-6,1e-12,mimo,2);


% to use the experimental bayesian updater
%[X,F,CP,History]  = AObayes(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],1e-3,1e-12,mimo,2);


%if ~mimo; [X,F,CP]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,DCM.xY.Q,1e-6,1e-12,0,2);   % MISO, curvature 
%else;     [X,F,CP]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,DCM.xY.Q,1e-6,1e-12,1,1);   % MIMO, gradients 
%end

%[X,F,CP]  = AOf(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],-inf);
[~,EP]    = fakeDM(X);

% return PP in input space
Pp = sum(diag(Pp)*cm');
Pp(Pp==0) = 1;


end

function [y,PP] = fakeDM(Px,varargin)
global DD

P    = DD.P;
cm   = DD.cm;

X0 = sum(diag(Px)*cm');
X0(X0==0) = 1;
X0 = full(X0.*exp(P'));
X0 = log(X0);
X0(isinf(X0)) = -1000;

PP = spm_unvec(X0,DD.SP);

IS   = spm_funcheck(DD.M.IS);       % Integrator
y    = IS(PP,DD.M,DD.xU);           % Prediction
y    = spm_vec(y);
y    = real(y);


end
