function [EP,F,CP] = AO_ERPDCM(P,G,DCM,niter,method)
% A wrapper for fitting DCMs with AO.m curvature optimisation routine.
% This version for ERP models. 
%
% Input parameter structure, P and a fully specified DCM structure:
%
%    [EP,F,CP] = AO_DCM(P,G,DCM)
%
% Prior to system identification, this function reformulates the problem as 
% a generalised model of the form
%   y = f(x,Pß)
% where AO.m calculates the coefficients, ß. These are then multipled back
% out to the actual parameter values.
%
% Returns posterior parameters [EP], squared error [F] and approximate
% covariance [CP]
%
% Dependencies: SPM
% AS
global DD

if nargin < 5 || isempty(method)
    method = 'fe';
end
if nargin < 4 || isempty(niter)
    niter = 128;
end

% put forward model priors into neural priors structure
% as per a spectral model
P.L    = G.L;
P.J    = G.J;
P.Lpos = G.Lpos;


DD    = DCM;
DD.SP = P;
P     = spm_vec(P);
V     = spm_vec(DCM.M.pC);
ip    = find(V);
cm    = zeros(length(V),length(ip));

% make and store a mapping matrix
for i = 1:length(ip)
    cm(ip(i),i) = 1;
end

% to pass to f(ßx)
DD.P  = P;
DD.V  = V;
DD.cm = cm;

fprintf('Performing AO optimisation\n');

p = ones(length(ip),1);
c = V(ip);

writelog = 0;

%[X,F,CP]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],1e-3,[]   ,0   ,2 ,writelog);
%[X,F,CP]  = AOf(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],-inf);

switch lower(method)
    case 'sse'
    % use this to minimise SSE:
    fprintf('Minimising SSE\n');
    [X,F,CP,History]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],1e-3,1e-12,2,writelog,'sse');
    case 'fe'
    % minimise free energy:
    fprintf('Minimising Free-Energy\n');
    [X,F,CP,History]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],-inf,1e-12,2,writelog,'fe',0,1,3);
    case 'logevidence'
    fprintf('Minimising -[log evidence]\n');
    [X,F,CP,History]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],-inf,1e-12,2,writelog,'logevidence');
end

[~,EP]    = fakeDM(X);


end

function [e,PP] = fakeDM(Px,varargin)
global DD

cm  = DD.cm;
P   = DD.P;                    % real world parameter vector
% hpE = DD.cm;                   % mapping from subspace to real
% x0  = (diag(Px) * hpE);        % indexing
% x0  = sum(x0);
% x0(x0==0) = 1;
% x0(isnan(x0)) = 1;
% 
% x1   = x0.*spm_vec(pE)';        % new (full) parameter values
X0 = sum(diag(Px)*cm');
X0(X0==0) = 1;
X0 = full(X0.*exp(P'));
X0 = log(X0);
X0(isinf(X0)) = -1000;

PP = spm_unvec(X0,DD.SP);

%PP   = spm_unvec(x1,pE);        % proper structure
M    = DD.M;
M.pC = ( spm_vec(DD.V)  );
M.pC = spm_unvec(M.pC,PP);


% f(x)
%--------------------------------------------------------------
[x,w] = feval(DD.M.IS,PP,M,DD.xU);

% g(x)
%--------------------------------------------------------------
L   = feval(DD.M.G , PP , DD.M);

% R
%--------------------------------------------------------------
R  = DD.xY.R;
M  = DD.M;
xY = DD.xY;
NNs = size(xY.y{1},1);
x0  = ones(NNs,1)*spm_vec(M.x)';         % expansion point for states

% FS(x-x0)
% FS(dxdp{i}*G',M)

for i = 1:length(x)
    y{i} = feval(DD.M.FS,(x{i}-x0)*L',DD.M);
end

% R
%--------------------------------------------------------------
R   = DD.xY.R;
M   = DD.M;
xY  = DD.xY;
Ns  = length(DD.A{1}); 
Nr  = size(DD.C,1); 
Nt  = length(x);

NNs = size(xY.y{1},1);
x0  = ones(NNs,1)*spm_vec(M.x)';         % expansion point for states
for i = 1:Nt
    K{i} = x{i} - x0;                   % centre on expansion point
    %y{i} = M.R*K{i}*L'*M.U;             % prediction (sensor space)
    %r{i} = M.R*xY.y{i}*M.U - y{i};      % residuals  (sensor space)
    %K{i} = K{i}(:,j);                   % Depolarization in sources
end

y = y';
e = y ;  % convert to residuals

end
