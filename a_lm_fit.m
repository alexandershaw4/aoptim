function  [beta,J,iter,cause,fullr,sse] = a_lm_fit(X,V,y,model,options,maxiter)
% Re-write of the Levenberg-Marquardt algorithm for dynamical model fitting
% to data...
%
%
% AS


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

if nargin < 6 || isempty(maxiter)
    maxiter = 32;
end

verbose = 3;

if verbose > 2
    fprintf('    Iteration    |   error      |   norm grad    |   norm step\n');
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

if nargin<8 || isempty(weights)
    sweights = ones(size(y));
else
    sweights = sqrt(weights);
end

% treatment for nans
yfit = model(beta,X,V);
fullr = sweights(:) .* (y(:) - yfit(:));
nans = isnan(fullr); % a col vector
r = fullr(~nans);
sse = (r'*r);% * (1 - corr(y(:), yfit(:)).^2);

%pp  = X;
%iS  = eye(length(r));
%sse = objective(r,iS,X,ones(length(X),length(r)),pp);

%vr = VtoGauss(r);
%sse = trace(vr*vr');

zerosp = zeros(p,1,class(r));
iter = 0;
breakOut = false;
cause = '';

if nargout>=2 && maxiter==0
    % Special case, no iterations but Jacobian needed
    J = getjacobian(beta,fdiffstep,model,X,yfit,nans,sweights,V);
end

while iter < maxiter
    iter = iter + 1;
    betaold = beta;
    sseold = sse;
    
    % Compute a finite difference approximation to the Jacobian
    J = getjacobian(beta,fdiffstep,model,X,yfit,nans,sweights,V);
    
    % Levenberg-Marquardt step: inv(J'*J+lambda*D)*J'*r
    diagJtJ = sum(abs(J).^2, 1);
    if funValCheck && ~all(isfinite(diagJtJ)), checkFunVals(J(:)); end
    Jplus = [J; diag(sqrt(lambda*diagJtJ))];
    rplus = [r; zerosp];
    step = Jplus \ rplus;

    %step = spm_dx(Jplus'*Jplus,rplus'*Jplus);
    %step = step(:);

    beta(:) = beta(:) + step;
    
    % Evaluate the fitted values at the new coefficients and
    % compute the residuals and the SSE.
    yfit = model(beta,X,V);
    fullr = sweights(:) .* (y(:) - yfit(:));
    r = fullr(~nans);
    sse = (r'*r) ;%* (1 - corr(y(:),yfit(:)).^2);

    % line search here?


    %sse = objective(r,iS,beta.*X,J',pp);

    %vr = VtoGauss(r);
    %sse = trace(vr*vr');

    if funValCheck && ~isfinite(sse), checkFunVals(r); end
    % If the LM step decreased the SSE, decrease lambda to downweight the
    % steepest descent direction.  Prevent underflowing to zero after many
    % successful steps; smaller than eps is effectively zero anyway.
    if sse < sseold
        lambda = max(0.1*lambda,eps);
        
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
            Jplus = [J; diag(sqrt(lambda*sum(J.^2,1)))];
            step = Jplus \ rplus;

            %step = spm_dx(Jplus'*Jplus,rplus'*Jplus);
            %step = step(:);

            beta(:) = betaold(:) + step;
            yfit = model(beta,X,V);
            fullr = sweights(:) .* (y(:) - yfit(:));
            r = fullr(~nans);
            sse = (r'*r) ;%* (1 - corr(y(:),yfit(:)).^2);

            %sse = objective(r,iS,beta.*X,J',pp);

            %vr = VtoGauss(r);
            %sse = trace(vr*vr');

            if funValCheck && ~isfinite(sse), checkFunVals(r); end
        end
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
        break
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
end % function LMfit

function sse = objective(r,iS,X,J,pp)

ny  = length(r);
Dg  = VtoGauss(real(r));
e   = trace(Dg*iS*Dg');
p   = ( X(:) - pp(:) );

Cp  = spm_inv(J*iS*J');
ipC = inv(Cp);

L(1) = spm_logdet(iS)*1/2  - e/2 - ny*log(8*atan(1))/2;
L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;

F    = sum(L);
sse    = (-F);

end





function checkFunVals(v)
% check if the function has finite output
if any(~isfinite(v))
    error(message('stats:nlinfit:NonFiniteFunOutput'));
end
end % function checkFunVals

function J = getjacobian(beta,fdiffstep,model,X,yfit,nans,sweights,V)
    %function yplus = call_model_nested(betaNew)
    %    yplus = model(betaNew, X,V);
    %    yplus(nans) = [];
    %end


    J = statjacobian(@(beta) model(beta,X), beta, fdiffstep, yfit(~nans),V);

%J = statjacobian(@call_model_nested, beta, fdiffstep, yfit(~nans),V);
if ~isempty(sweights)
    sweights = sweights(~nans);
    J = bsxfun(@times,sweights(:),J);
end
end % function getjacobian

function J = statjacobian(func, theta, DerivStep, y0,V)
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
