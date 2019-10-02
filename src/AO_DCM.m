function [EP,F,CP] = AO_DCM(P,DCM,niter)
% A wrapper for fitting DCMs with AO.m curvature optimisation routine.
% 
% Input parameter structure, P and a fully specified DCM structure:
%
%    [EP,F,CP] = AO_DCM(P,DCM)
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

if nargin < 3 || isempty(niter)
    niter = 128;
end

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

[X,F,CP]  = AO(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],1e-3);
%[X,F,CP]  = AOf(@fakeDM,p(:),c,DCM.xY.y,niter,12*4,[],-inf);
[~,EP]    = fakeDM(X);


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
