function [j,ip,j1] = jaco_mimo_par(fun,x0,V,verbose,order,nout,params)
% Compute the 1st or 2nd order partial numerical derivates of a function
% - parameter version: i.e. dp/dx - using symmetric finite difference methods
%
% - this version (jaco_mimo) is a variant on jaco (jaco.m) that computes
% the partial derivatives of a multiple output function by catching all
% outputs of the function into a cell and vectorising them - i.e. for a
% model of the form:
%
%     [y1,y2] = fun(p),  it would compute dpdy1 and dpdy2 simultaneously
%
% usage: [j,ip] = jaco(fun,x0,V,verbose,order,nout,{params})
%
%  fun = function
%  x0  = params, e.g. fun(x0)
%  V   = step for each param (e.g. 1/8 ... ) [can be vector: 1 for each x0]
%  verbose = print progress
%  order = method/derivate order, see below
%  nout = number of outputs of the function to diff w r t
%  params = any req. additional function inputs in cell, e.g. fun(x0,params{:})
%
% Order/options flags:
% ----------------------------------------------------------------------
% (1.) 1st order numerical derivatives (gradient) 
% of a function using:
%   j(ip,:) = ( f(x(ip)+h)  - f(x(ip)-h) )  / (2 * h)
%
% (2.) 2nd order numerical derivatives (curvature):
%   j(ip,:) = [ (f0 - f1) / 2 / d ] ./ [ (f0 - 2 * fx + f1) / d ^ 2 ]
%
% (0.) One-sided 1st order approximation (quick):
%   j(ip,:) = ( f(x(ip)+h) - fx ) / h
%
% (-1:) Complex conjugate curvature method:
%   f0(ip)  = ( f(x(ip) + h * 1i) - fx ) / h
%   d1      = ( real(f0) - imag(f0) ) / 2 / d;
%   d2      = ( real(f0) - 2 * fx + imag(f0) ) / d ^ 2;
%   j(i,:)  = d1 ./ d2;
%
% (3.) 1st order 'central' method:
%   j(i,:) = ( ( (f0 - fx) / (2*d) ) + ( (fx - f1) / (2*d) ) ) ./2;
%
% (4.) 2nd order 'central' method:
%   d1a    = (f0 - fx) / (2*d);
%   d1b    = (fx - f1) / (2*d);
%   d2a    = (f0 - 2 * fx + f1) / d ^ 2;
%   j(i,:) = ( (d1a + d1b)./2 ) ./ d2a;
%
% (5.) Higher-order: 5-point method
%   j(i,:) = ( (-f0 + (8*f1) - (8*f2) + f3 ) ./ 12*d ) ;%+ ...
%                      % ( ((d.^4)/30)*fc );
%
% (007.) - see below.
%
% For systems of the form dx = f(x):
%  if order==1, when j is square, it is the Jacobian
%  if order==2, when j is square, it is the Hessian
% 
% AS2019

if nargin < 7 || isempty(params) ; params  = []; end
if nargin < 6 || isempty(nout)   ; nout    = 1;  end
if nargin < 5 || isempty(order)  ; order   = 1; end
if nargin < 4 || isempty(verbose); verbose = 0; end

% initialise output cells
op = cell(1,nout); %need to finish this - for derivs of a MIMO - i.e.
% w.r.t to a multiple output function 

if isempty(params)
    IS = fun;
    P  = x0(:);
else
    IS = @(P) fun(P,params{:});
    P  = x0(:) ;
end

if nargin >= 3; ip = ~~(V(:));
else;           ip = 1:length(x0);
end

% The subfunction -
warning off; % suppress matrix singularity warnings in unstable systems
[j,j1]  = jacf(IS,P,ip,verbose,V,order,nout);
warning on;

if isnumeric(j)
    j(isnan(j)) = 0;
end
if iscell(j)
    for i = 1:length(j)
        j{i}(isnan(j{i})) = 0;
    end
end

%j(isinf(j)) = 0;

end



function [j,j1] = jacf(IS,P,ip,verbose,V,order,nout)

% Compute the Jacobian matrix using variable step-size
n  = 0;
%j1 = [];
%warning off ;

if verbose
    switch order
        case 1 ; fprintf('Copmuting 1st order pd (Gradient/Jacobian)\n');
        case 2 ; fprintf('Computing 2nd order pd (Curvature)\n');
    end
end

%initialise things
f0{1} = cell(1,nout);
[f0{1}{:}] = feval(IS,P);
fx = f0{1};
f1 = f0;

f0 = repmat(f0,[length(P),1]);
f1 = f0;
j = cell(length(P),nout);
j1=j;
ff = f1;


if ismember(order,[1 2 3 4])
    parfor i = 1:length(P)
        if ip(i)

            % Compute Jacobi: A(j,:) = ( f(x+h) - f(x-h) ) / (2 * h)
            P0     = P;
            P1     = P;
            d      = P0(i) * V(i);

            if d == 0
                d = 0.01;
            end

            P0(i)  = P0(i) + d  ;
            P1(i)  = P1(i) - d  ;

            f0{i} = cell(1,nout);
            f1{i} = cell(1,nout);
            
            [f0{i}{:}] = feval(IS,P0);
            [f1{i}{:}] = feval(IS,P1);
            j(i,:) = spm_unvec( ( spm_vec(f0{i}) - spm_vec(f1{i}) ) / (2 * d) , fx );
            
            
            if order == 3 || order == 4
                       
                j(i,:) = spm_unvec( ((spm_vec(f0{i}) - spm_vec(fx{i})) ./ (2*d) ) + ...        
                                    ((spm_vec(fx{i}) - spm_vec(f1{i})) ./ (2*d) ) ./2 , fx);  
                
            end
            
            j1(i,:) = j(i,:);
            
            if order == 2 
                j1(i,:) = j(i,:); % keep first order
                                
                deriv1 = spm_unvec( ( spm_vec(f0{i}) - spm_vec(f1{i}) ) / 2 / d , fx );
                deriv2 = spm_unvec( (spm_vec(f0{i}) - 2 * spm_vec(fx) + spm_vec(f1{i})) / d.^2 , fx);
                j(i,:) = spm_unvec( spm_vec(deriv1)./spm_vec(deriv2) , fx);
                
            elseif order == 4
                % curvature using the 3-point routine
                deriv1a = spm_unvec( (spm_vec(f0{i}) - spm_vec(fx)) / (2*d), fx);
                deriv1b = spm_unvec( (spm_vec(fx) - spm_vec(f1{i})) / (2*d), fx);
                deriv2a = spm_unvec( (spm_vec(f0{i}) - 2 * spm_vec(fx) + spm_vec(f1{i})) / d ^ 2, fx);
                j(i,:)  = spm_unvec( ( (spm_vec(deriv1a) + spm_vec(deriv1b))./2 ) ./ spm_vec(deriv2a), fx);
                
            end
        else
            j(i,:) = spm_unvec(spm_vec(ff{1})*0,ff{1});
            j1(i,:) = j(i,:);
        end
    end
    

elseif ismember(order,22)
    
    k = 3;
    
    % Belyaev 1999 curvature
    %-------------------------------
    for i = 1:length(P)
        if ip(i)
            
            P0     = P;
            P1     = P;
            a      = P0(i) * V(i);
            b      = P0(i) * V(i) * k;

            if a == 0
                a = 0.01;
                b = a * k;
            end

            P0(i)  = P0(i) - a  ;
            P1(i)  = P1(i) + b  ;

            % f''(x) =  2f(x-a)       2f(x)     2f(x+b)
            %          ---------  -  ------- +  -------
            %            a(a+b)         ab       b(a+b)


            f0     = spm_vec(spm_cat(feval(IS,P0)));    
            f1     = spm_vec(spm_cat(feval(IS,P1)));

            j(i,:) = ( 2*f0 ./ (a*(a+b)) ) - ...
                     ( 2*fx ./ (a*b)     ) + ...
                     ( 2*f1 ./ (b*(a+b)) );
        end
    end
    
elseif ismember(order,-2)
    
    % 0 order diff, i.e. f(x+d) - f(x) / d
    % AND curvature
    for i = 1:length(P)
            if ip(i)
                P0     = P;
                P1     = P;
               %d      = P0(i) * V(i);
                d      =         V(i);

                if d == 0;d = 0.01;end

                P0(i)  = P0(i) + d  ;
                P1(i)  = P1(i) - d  ;
                
                f0     = spm_vec(spm_cat(feval(IS,P0)));
                f1     = spm_vec(spm_cat(feval(IS,P1)));
                
                j(i,:) = (f0 - fx) / P0(i);%(d);
                
                % curvs
                deriv1 = (f0 - f1) / 2 / d;
                deriv2 = (f0 - 2 * fx + f1) / d ^ 2;
                c(i,:) = deriv1 ./ deriv2;
            end
    end
    
    % in this special useage case, return both in cell array
    j = {j c};
    
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
            j(i,:) = ( (-f0 + (8*f1) - (8*f2) + f3 ) ./ 12*d ) ;%+ ...
                       % ( ((d.^4)/30)*fc );


        end
    end
    

elseif ismember(order,0)
    clear f0 f1
    f0 = cell(1,nout);;
    [f0{1}{:}] = feval(IS,P);
    
    % 0 order diff, i.e. f(x+d) - f(x) / d
    % this is a cheap approximation but requires half the number of
    % function evaluations that order 1&2 would...
    parfor i = 1:length(P)
            if ip(i)
                P0     = P;
                %d      = P0(i) * V(i);
                d = V(i);

                if d == 0;d = 0.01;end

                P0(i)  = P0(i) + d  ;
                
                f1{i} = cell(1,nout);
                
                [f1{i}{:}] = feval(IS,P0);
                
                %j(i,:) = ( (spm_vec(f0) - spm_vec(fx)) / (d) );
                
                j(i,:) = spm_unvec( ( spm_vec(f0{i}) - spm_vec(f1{i}) ) / (2 * d) , {fx} );
                
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

%warning on;
if verbose
    fprintf('\n');
end

end