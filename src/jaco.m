function [j,ip] = jaco(fun,x0,V,verbose,order)
% Compute the 1st or 2nd order partial (numerical) derivates of a function
% - parameter version: i.e. dp/dx - using symmetric finite difference
%
% usage: [j,ip] = jaco(fun,x0,V,verbose,order)
%
% Orders:
% --------------------------------------------------------
% (order 1:) Compute the 1st order partial derivatives (gradient) 
% of a function using:
%
% j(ip,:) = ( f(x(ip)+h)  - f(x(ip)-h) )  / (2 * h)
%
% (order 2:) Compute the 2nd order derivatives (curvature):
%
% j(ip,:) = [ (f0 - f1) / 2 / d ] ./ [ (f0 - 2 * fx + f1) / d ^ 2 ]
%
% (order 0:) One-sided 1st order approximation (quick)
%
% j(ip,:) = ( f(x(ip)+h) - fx ) / h
%
% (order -1:) Complex conjugate curvature method:
%
% f0(ip)  = ( f(x(ip) + h * 1i) - fx ) / h
% d1      = ( real(f0) - imag(f0) ) / 2 / d;
% d2      = ( real(f0) - 2 * fx + imag(f0) ) / d ^ 2;
% j(i,:)  = d1 ./ d2;
%
% For systems of the form dx = f(x):
% if order==1, when j is square, it is the Jacobian
% if order==2, when j is square, it is the Hessian
% 
% AS2019

if nargin < 5 || isempty(order)
    order = 1;
end

if nargin < 4 || isempty(verbose)
    verbose = 0;
end

IS = fun;
P  = x0(:);

% if nargin == 3; ip = find(V(:));
% else;           ip = 1:length(x0);
% end

if nargin >= 3; ip = ~~(V(:));
else;           ip = 1:length(x0);
end

j  = jacf(IS,P,ip,verbose,V,order);

j(isnan(j)) = 0;
%j(isinf(j)) = 0;

end



function j = jacf(IS,P,ip,verbose,V,order)

% Compute the Jacobian matrix using variable step-size
n  = 0;
warning off ;

if verbose
    switch order
        case 1 ; fprintf('Copmuting 1st order pd (Gradient/Jacobian)\n');
        case 2 ; fprintf('Computing 2nd order pd (Curvature)\n');
    end
end

%f0    = feval(IS,P);
f0    = spm_cat( feval(IS,P) );
fx    = f0(:);
j     = zeros(length(P),length(f0(:))); % n param x n output
if ismember(order,[1 2])
    for i = 1:length(P)
        if ip(i)

            % Print progress
            n = n + 1;
            if verbose
                if n > 1; fprintf(repmat('\b',[1,length(str)])); end
                str  = sprintf('Computing Gradients [ip %d / %d]',n,length(find(ip)));
                fprintf(str);
            end

            % Compute Jacobi: A(j,:) = ( f(x+h) - f(x-h) ) / (2 * h)
            P0     = P;
            P1     = P;
            d      = P0(i) * V(i);

            if d == 0
                d = 0.01;
            end

            P0(i)  = P0(i) + d  ;
            P1(i)  = P1(i) - d  ;

            f0     = spm_vec(spm_cat(feval(IS,P0)));
            f1     = spm_vec(spm_cat(feval(IS,P1)));
            j(i,:) = (f0 - f1) / (2 * d);

            if order == 2
                % Alternatively, include curvature
                deriv1 = (f0 - f1) / 2 / d;
                deriv2 = (f0 - 2 * fx + f1) / d ^ 2;
                j(i,:) = deriv1 ./ deriv2;
            end
        end
    end
    
elseif ismember(order,5)
    
    % Higher order method:
    % five-point method - not great
    for i = 1:length(P)
        if ip(i)

            % Print progress
            n = n + 1;
            if verbose
                if n > 1; fprintf(repmat('\b',[1,length(str)])); end
                str  = sprintf('Computing Gradients [ip %d / %d]',n,length(find(ip)));
                fprintf(str);
            end

            % Compute components
            P0     = P;
            P1     = P;
            P2     = P;
            P3     = P;
            Pc     = P;
            d      = P0(i) * V(i);

            if d == 0
                d = 0.01;
            end

            P0(i)  = P0(i) + (d*2)  ;
            P1(i)  = P1(i) + (d*1)  ;
            P2(i)  = P2(i) - (d*1)  ;
            P3(i)  = P3(i) - (d*2)  ;
            Pc(i)  = Pc(i) + (d/8);

            f0     = spm_vec(spm_cat(feval(IS,P0)));
            f1     = spm_vec(spm_cat(feval(IS,P1)));
            f2     = spm_vec(spm_cat(feval(IS,P2)));
            f3     = spm_vec(spm_cat(feval(IS,P3)));
            fc     = spm_vec(spm_cat(feval(IS,Pc)));
            
            % full formula
            j(i,:) = ( (-f0 + 8*f1 - 8*f2 + f3 ) ./ 12*d ) + ...
                        ( ((d.^4)/30)*fc );


        end
    end
    
elseif ismember(order,0)
    
    % 0 order diff, i.e. f(x+d) - f(x) / d
    % this is a cheap approximation but requires half the number of
    % function evaluations that order 1&2 would...
    for i = 1:length(P)
            if ip(i)
                P0     = P;
                d      = P0(i) * V(i);

                if d == 0;d = 0.01;end

                P0(i)  = P0(i) + d  ;
                f0     = spm_vec(spm_cat(feval(IS,P0)));
                j(i,:) = (f0 - fx) / (d);
            end
    end

elseif ismember(order,-1)
    
    % complex conjugate gradient method
    % d(i,:) = imag(f(x+d*i1))/d
    
    for i = 1:length(P)
            if ip(i)
                P0     = P;
                d      = P0(i) * V(i);

                if d == 0;d = 0.01;end

                P0(i)  = P0(i) + d * 1i  ;
                f0     = spm_vec(spm_cat(feval(IS,P0)));
                j(i,:) = imag(f0) / d;
                
                deriv1 = ( real(f0) - imag(f0) ) / 2 / d;
                deriv2 = ( real(f0) - 2 * fx + imag(f0) ) / d ^ 2;
                j(i,:) = deriv1 ./ deriv2;
            end
    end
    
elseif ismember(order,007)
    
    theta = P;
    func  = @(x) spm_vec(spm_cat(feval(IS,x)));
    y0    = fx;
    DerivStep = eps^(1/3);
    
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
    
    func = @(theta) func(reshape(theta, thetaOriginalSize));
    
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
        % Calculate delta(:,ii), but remember to set it back to 0 at the end of the loop.
        delta(:,ii) = DerivStep(ii) * theta(:,ii);
        deltaZero = delta(:,ii) == 0;
        if any(deltaZero)
            % Use the norm as the "scale", or 1 if the norm is 0.
            nTheta = sqrt(sum(theta(deltaZero,:).^2, 2));
            delta(deltaZero,ii) = DerivStep(ii) * (nTheta + (nTheta==0));
        end
        thetaNew = theta + delta;
        yplus = func(thetaNew);
        dy = yplus(:) - y0(:);
        J(:,ii) = dy./delta(rowIdx,ii);
        delta(:,ii) = 0;
    end
    
    
end

warning on;
if verbose
    fprintf('\n');
end

end