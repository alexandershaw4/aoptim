function J = getjacobian(beta,fdiffstep,model,yfit,sweights,V,UseParallel)
    %function yplus = call_model_nested(betaNew)
    %    yplus = model(betaNew, X,V);
    %    yplus(nans) = [];
    %end


    J = statjacobian(@(beta) model(beta), beta, fdiffstep, yfit,V,UseParallel);

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
