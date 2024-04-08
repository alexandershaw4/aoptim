function  [beta,J,iter,cause,fullr,sse] = a_lm_VL(X,V,y,model,options,maxiter,usepar,Q)
% Returns maximum aposteriori estimates (MAP) for the parameters of the
% nonlinear 'model' given initial parameters 'X', variances 'V' and target data
% to fit 'y'. Uses Gauss-Newton & Levenberg-Marquardt steps, objective function 
% is the free energy / ELBO; this version approximates Variational Laplace
%
% [beta,J,iter,cause,fullr,sse] = a_lm_VL(X,V,y,model,options,maxiter,usepar)
%
% AS2024


beta  = ones(size(X));
doplot = 1;

if doplot
    figure;
end

if nargin < 5 || isempty(options)
    options.Display= 'iter';
    options.MaxFunEvals= [];
    options.MaxIter= 32;
    options.TolBnd= [];
    options.TolFun= 1.0000e-06;
    options.TolTypeFun= [];
    options.TolX= 1.0000e-08;
    options.TolTypeX= [];
    options.GradObj= [];
    options.Jacobian= [];
    options.DerivStep= 1/8;%e-8;
    options.FunValCheck= 'on';
    options.Robust= 'off';
    options.RobustWgtFun= [];
    options.WgtFun= 'bisquare';
    options.Tune= [];
    options.UseParallel= [];
    options.UseSubstreams= [];
    options.Streams= {};
    options.OutputFcn= [];
end

if nargin == 7 && usepar
    usepar=1;
else 
    usepar=0; % still need to implement to parallel version of jacobian routine
end

if nargin < 6 || isempty(maxiter)
    maxiter = 32;
end

verbose = 3;

if verbose > 2
    fprintf('    Iteration    |   objective  |   norm grad    |   norm step\n');
end

% Set up convergence tolerances from options.
betatol = options.TolX;
rtol = options.TolFun;
fdiffstep = options.DerivStep;
if isscalar(fdiffstep)
    fdiffstep = repmat(fdiffstep, size(beta));
else
    % statset ensures fdiffstep is not a matrix.
    % Here, we ensure fdiffstep has the same shape as beta.
    fdiffstep = reshape(fdiffstep, size(beta));
end
funValCheck = strcmp(options.FunValCheck, 'on');

% Set initial weight for LM algorithm.
lambda = .01;

% Set the iteration step
sqrteps = sqrt(eps(class(beta)));

p = numel(beta);

%if nargin<8 || isempty(weights)
    sweights = ones(size(y));
%else
%    sweights = sqrt(weights);
%end

% treatment for nans
yfit = model(beta,X,V);
fullr = sweights(:) .* (y(:) - yfit(:));
nans = isnan(fullr); % a col vector
r = fullr(~nans);
%sse = (r'*r);% * (1 - corr(y(:), yfit(:)).^2);

% initialise aopt structure for objective
pp  = X;
aopt.Q = [];
aopt.ipC = denan(diag(1./V),1/8);

if nargin == 8 && ~isempty(Q)
    aopt.Q = Q;
end

[sse,aopt] = objective(r,V,X,ones(length(X),length(r)),pp,y(:),aopt);

zerosp = zeros(p,1,class(r));
iter = 0;
breakOut = false;
cause = '';

w = 1:length(y);

% switch for Jacobian regularisation when step size too small
RegJac = 0;

if nargout>=2 && maxiter==0
    % Special case, no iterations but Jacobian needed
    J = getjacobian(beta,fdiffstep,model,X,yfit,nans,sweights,V,usepar);
end

while iter < maxiter
    iter = iter + 1;
    betaold = beta;
    sseold = sse;
    
    % Compute a finite difference approximation to the Jacobian
    J = getjacobian(beta,fdiffstep,model,X,yfit,nans,sweights,V,usepar);
    %J = jaco_gauss(@(beta) model(beta,X,V),beta); J = J';
    
    % Levenberg-Marquardt step: inv(J'*J+lambda*D)*J'*r
    diagJtJ = sum(abs(J).^2, 1);
    Cp  = diag( sqrt(lambda*sum(J'*aopt.iS*J,1)) );
    if funValCheck && ~all(isfinite(diagJtJ)), checkFunVals(J(:)); end
    %Jplus = [J*aopt.ipC; diag(sqrt(lambda*diagJtJ))]; 
    Jplus = [J; Cp];
    rplus = [r; zerosp];
    %step = Jplus \ rplus;
    step =  atcm.fun.aregress(Jplus,rplus,'MAP');
    beta(:) = beta(:) + step;
    
    % Evaluate the fitted values at the new coefficients and
    % compute the residuals and the SSE.
    yfit = model(beta,X,V);
    fullr = sweights(:) .* (y(:) - yfit(:));
    r = fullr(~nans);
    %sse = (r'*r) ;

    [sse,n_aopt] = objective(r,V,X,J',pp,y(:),aopt);
    if funValCheck && ~isfinite(sse), checkFunVals(r); end
    % If the LM step decreased the SSE, decrease lambda to downweight the
    % steepest descent direction.  Prevent underflowing to zero after many
    % successful steps; smaller than eps is effectively zero anyway.
    if sse < sseold
        lambda = max(0.1*lambda,eps);
        
        aopt = n_aopt;

        % If the LM step increased the SSE, repeatedly increase lambda to
        % upweight the steepest descent direction and decrease the step size
        % until we get a step that does decrease SSE.
    else
        while sse > sseold
            lambda = 10*lambda;
            if lambda > 1e16
                breakOut = true;
                break
            end
            Cp  = diag( sqrt(lambda*sum(J'*aopt.iS*J,1)) );
            %Jplus = [J*aopt.ipC; diag(sqrt(lambda*sum(J.^2,1)))];
            Jplus = [J; Cp];
            %step = Jplus \ rplus;
            step =  atcm.fun.aregress(Jplus,rplus,'MAP');

            beta(:) = betaold(:) + step;
            yfit = model(beta,X,V);
            fullr = sweights(:) .* (y(:) - yfit(:));
            r = fullr(~nans);
            %sse = (r'*r) ;

            [sse,n_aopt] = objective(r,V,X,J',pp,y(:),aopt);

            if funValCheck && ~isfinite(sse), checkFunVals(r); end
        end
        aopt = n_aopt;
    end
    if verbose > 2 % iter
        disp(sprintf('      %6d    %12g    %12g    %12g', ...
            iter,sse,norm(2*r'*J),norm(step))); %#ok<DSPS>
    end

    alle(iter) = sse;
    if doplot
        tw = 1:length(y);
        subplot(211); plot(tw,y,tw,yfit);

        subplot(212); plot(alle','*');hold on; plot(alle); hold off;drawnow;

    end
    
    % Check step size and change in SSE for convergence.
    if norm(step) < betatol*(sqrteps+norm(beta))
        cause = 'tolx';
        cause
        %break
        %fprintf('Regularising gradients (hit %s)\n',cause);
        %RegJac = 1;

    elseif abs(sse-sseold) <= rtol*sse
        cause = 'tolfun';
        cause
        break
    elseif breakOut
        cause = 'stall';
        cause
        break
    end
end
if (iter >= maxiter)
    cause = 'maxiter';
end
end 

function [e,aopt] = objective(r,V,X,J,pp,data,aopt)

y = data - r(:); Y = data; x0 = X;

% dgY = VtoGauss(real(Y));
% dgy = VtoGauss(real(y));
% 
% Dg  = dgY - dgy;
% e   = trace(Dg*iS*Dg');
% 
% % and scaled version
% dgYn = VtoGauss(real(Y./sum(Y)));
% dgyn = VtoGauss(real(y./sum(y)));
% 
% Dgn  = dgYn - dgyn;
% en   = trace(Dgn*iS*Dgn');
% 
% e = e + 8*en;

% end accuracy of model


% Free Energy Objective Function: F(p) = log evidence - divergence
%--------------------------------------------------------------------------
Q  = aopt.Q;

if isnumeric(Q) && ~isempty(Q) 
    % If user supplied a precision matrix, store it so that it can be
    % incorporated into the updating q
    aopt.precisionQ = Q;
elseif iscell(Q)
    aopt.precisionQ = Q;
end

if ~isfield(aopt,'precisionQ')
    Q  = spm_Ce(1*ones(1,length(spm_vec(y)))); %
    ny  = length(spm_vec(y));
    nq  = ny ./ length(Q);
elseif isfield(aopt,'precisionQ') && isnumeric(aopt.precisionQ)
    Q   = {aopt.precisionQ};
    clear Q;
    lpq = length(aopt.precisionQ);
    for ijq = 1:length(aopt.precisionQ)
       Q{ijq} = sparse(ijq,ijq,aopt.precisionQ(ijq,ijq),lpq,lpq);
    end

    ny  = length(spm_vec(y));
    nq  = ny ./ length(Q{1});
elseif isfield(aopt,'precisionQ') && iscell(aopt.precisionQ)
    Q = aopt.precisionQ;
    ny  = length(spm_vec(y));
    nq  = ny ./ length(Q{1});
end

if ~isfield(aopt,'h') 
    h  = sparse(length(Q),1) - log(var(spm_vec(Y))) + 4;
else
    h = aopt.h;
end

if any(isinf(h))
    h = denan(h)+1/8;
end

iS = sparse(0);

for i  = 1:length(Q)
    iS = iS + Q{i}*(exp(-32) + exp(h(i)));
end

e   = (spm_vec(Y) - spm_vec(y)).^2;
ipC = aopt.ipC;

warning off;                                % suppress singularity warnings
Cp  = spm_inv( (J*iS*J') + ipC );
%Cp = (Cp + Cp')./2;
warning on

p  = ( x0(:) - pp(:) );

if any(isnan(Cp(:))) 
    Cp = denan(Cp,1/8);
end


% pulled directly from SPM's spm_nlsi_GN.m ...
% ascent on h / precision {M-step}
%==========================================================================
%for m = 1:8
clear P;
nh  = length(Q);
warning off;
S   = spm_inv(iS);warning on;

ihC = speye(nh,nh)*exp(4);
hE  = sparse(nh,1) - log(var(spm_vec(Y))) + 4;
for i = 1:nh
    P{i}   = Q{i}*exp(h(i));
    PS{i}  = P{i}*S;
    P{i}   = kron(speye(nq),P{i});
    JPJ{i} = real(J*P{i}*J');
end

% derivatives: dLdh
%------------------------------------------------------------------
for i = 1:nh
    dFdh(i,1)      =   trace(PS{i})*nq/2 ...
        - real(e'*P{i}*e)/2 ...
        - spm_trace(Cp,JPJ{i})/2;
    for j = i:nh
        dFdhh(i,j) = - spm_trace(PS{i},PS{j})*nq/2;
        dFdhh(j,i) =   dFdhh(i,j);
    end
end

% add hyperpriors
%------------------------------------------------------------------
d     = h     - hE;
dFdh  = dFdh  - ihC*d;
dFdhh = dFdhh - ihC;
Ch    = spm_inv(-dFdhh);

% update ReML estimate
%------------------------------------------------------------------
warning off;
dh    = spm_dx(dFdhh,dFdh,{4});
dh    = min(max(dh,-1),1);
warning on;
h     = h  + dh;

%if aopt.updateh
    aopt.h = h;
    aopt.JPJ = JPJ;
    aopt.Ch  = Ch;
    aopt.d   = d;
    aopt.ihC = ihC;
    aopt.iS = iS;
%end


% compute accuracy of model
dgY = VtoGauss(real(Y));
dgy = VtoGauss(real(y));

Dg  = dgY - dgy;
e   = trace(Dg*iS*Dg');

% and scaled version
dgYn = VtoGauss(real(Y./sum(Y)));
dgyn = VtoGauss(real(y./sum(y)));

Dgn  = dgYn - dgyn;
en   = trace(Dgn*iS*Dgn');

e = e + 8*en;


% Compute objective function;

L(1) = spm_logdet(iS)*nq/2  - e/2 - ny*log(8*atan(1))/2;
%L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;
L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;

F    = sum(L);
e    = (-F);






% model = data - r(:);
% 
% Vmodel = VtoGauss(model);
% Vdata  = VtoGauss(data);
% e      = mvgkl(data,Vdata,model,Vmodel);
% %e = e + mvgkl(model,Vmodel,data,Vdata);;
% 
% % Parameters;
% 
% for i = 1:length(pp)
%     pd(i)  = makedist('normal','mu', pp(i),'sigma', V(i) );
% end
% 
% % Compute relative change in cdf
% f   = @(dx,pd) (1./(1+exp(-pdf(pd,dx)))) ./ (1./(1+exp(-pdf(pd,pd.mu))));
% 
% for i = 1:length(X)    
%     PX(i) = f(X(i),pd(i));
% end
% 
% px = 1 - (PX*J)*data; %sum(PX);%1 - (PX*J)'\data;
% 
% e = e * px;

end

% function sse = objective(r,iS,X,J,pp)
% 
% ny  = length(r);
% Dg  = VtoGauss(real(r));
% e   = trace(Dg*iS*Dg');
% p   = ( X(:) - pp(:) );
% 
% Cp  = spm_inv(J*iS*J');
% ipC = inv(Cp);
% 
% L(1) = spm_logdet(iS)*1/2  - e/2 - ny*log(8*atan(1))/2;
% L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;
% 
% F    = sum(L);
% sse    = (-F);
% 
% end





function checkFunVals(v)
% check if the function has finite output
if any(~isfinite(v))
    error(message('stats:nlinfit:NonFiniteFunOutput'));
end
end % function checkFunVals

function J = getjacobian(beta,fdiffstep,model,X,yfit,nans,sweights,V,UseParallel)
    %function yplus = call_model_nested(betaNew)
    %    yplus = model(betaNew, X,V);
    %    yplus(nans) = [];
    %end


    J = statjacobian(@(beta) model(beta,X), beta, fdiffstep, yfit(~nans),V,UseParallel);

%J = statjacobian(@call_model_nested, beta, fdiffstep, yfit(~nans),V);
if ~isempty(sweights)
    sweights = sweights(~nans);
    J = bsxfun(@times,sweights(:),J);
end
end % function getjacobian

function J = statjacobian(func, theta, DerivStep, y0,V,UseParallel)
%STATJACOBIAN Estimate the Jacobian of a function

% J is a matrix with one row per observation and one column per model
% parameter. J(i,j) is an estimate of the derivative of the i'th
% observation with respect to the j'th parameter.

% For performance reasons, very little error checking is done on the input
% arguments. This function makes the following assumptions about inputs:
%
% * func is the model function and is a valid function handle that accepts
%   a single input argument of the same size as theta.
% * theta is vector or matrix of parameter values. If a matrix, each row
%   represents a different group or observation (see "Grouping Note" below)
%   and each column represents a different model parameter.
% * DerivStep (optional) controls the finite differencing step size. It may
%   be empty, scalar, or a vector of positive numbers with the number of
%   elements equal to the number model parameters.
% * y0 (optional) is the model function evaluated at theta. A value of []
%   is equivalent to omitting the argument and results in the model being
%   evaluated one additional time.
%
% Example 1: NLINFIT
%   NLINFIT is used to estimate the parameters b(1) and b(2) for the model
%   @(b,T) b(1)*sin(b(2)*T), given data at T=1:5. NLINFIT needs the
%   Jacobian of the model function with respect to b(1) and b(2) at each T.
%   To do this, it constructs a new function handle that is only a function
%   of b and that "burns-in" the value of T (e.g. model2 = @(b) model1(b,T)).
%   It then calls STATJACOBIAN with the new function handle to obtain a
%   matrix J, where J(i,j) is an estimate of the derivative of the model
%   with respect to the j'th parameter evaluated at T(i) and b.
%
% Example 2: NLMEFIT or NLMEFITSA with group-specific parameters
%   NLMEFIT requires the Jacobian of the model function with respect to two
%   parameters evaluated at group-specific values. (Group-specific
%   parameters can arise, for example, from using the default FEConstDesign
%   and REConstDesign options.) NLMEFIT calls STATJACOBIAN passing in a
%   matrix of parameter values theta, with one row per group, where
%   theta(i,j) represents a parameter value for i'th group and j'th
%   parameter. STATJACOBIAN returns a matrix J, where J(i,j) is an estimate
%   of the derivative of the model with respect to the j'th parameter,
%   evaluated for observation i with parameter values theta(rowIdx(i),:),
%   which are the parameter values for the observation's group.
%
% Example 3: NLMEFIT with observation-specific parameters
%   NLMEFIT requires the Jacobian of the model function with respect to two
%   parameters evaluated at observation-specific values. (Observation-
%   specific parameters can arise, for example, from using the FEObsDesign
%   or REObsDesign options.) NLMEFIT calls STATJACOBIAN passing in a matrix
%   of parameter values theta, with one row per observation, where
%   theta(i,j) represents a parameter value for the i'th observation and
%   j'th parameter. In this case, rowIdx is 1:N, where N is the number of
%   observations. STATJACOBIAN returns a matrix J, where J(i,j) is an
%   estimate of the derivative of the model with respect to the j'th
%   parameter, evaluated for observation i with parameter values
%   theta(i,:), which are the parameter values for the observation.

% Use the appropriate class for variables.
classname = class(theta);

% Handle optional arguments, starting with y0 since it will be needed to
% determine the appropriate size for a default groups.
if nargin < 4 || isempty(y0)
    y0 = func(theta);
end

% When there is only one group, ensure that theta is a row vector so
% that vectoriation works properly. Also ensure that the underlying
% function is called with an input with the original size of theta.
thetaOriginalSize = size(theta);
theta = reshape(theta, 1, []);

funct = func;% @(theta) func(reshape(theta, thetaOriginalSize));

% All observations belong to a single group; scalar expansion allows us
% to vectorize using a scalar index.
rowIdx = 1;

[numThetaRows, numParams] = size(theta);

if nargin < 3 || isempty(DerivStep)
    % Best practice for forward/backward differences:
    DerivStep = repmat(sqrt(eps(classname)), 1, numParams);
    % However, NLINFIT's default is eps^(1/3).
elseif isscalar(DerivStep)
    DerivStep = repmat(DerivStep, 1, numParams);
end

delta = zeros(numThetaRows, numParams, classname);
J = zeros(numel(y0), numParams, classname);
    for ii = 1:numParams
        if ~~full(V(ii))
            % Calculate delta(:,ii), but remember to set it back to 0 at the end of the loop.
            delta(:,ii) = DerivStep(ii) * theta(:,ii);
            deltaZero = delta(:,ii) == 0;
            if any(deltaZero)
                % Use the norm as the "scale", or 1 if the norm is 0.
                nTheta = sqrt(sum(theta(deltaZero,:).^2, 2));
                delta(deltaZero,ii) = DerivStep(ii) * (nTheta + (nTheta==0));
            end
            thetaNew = theta + delta;
            yplus = funct(thetaNew);
            dy = yplus(:) - y0(:);
            J(:,ii) = dy./delta(rowIdx,ii);
            delta(:,ii) = 0;
        else
            J(:,ii) = 0;
        end
    end

end
