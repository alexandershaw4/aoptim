function [j,ip] = jaco(fun,x0,V,verbose,order)
% Compute the 1st or 2nd order partial (numerical) derivates of a function
% - parameter version: i.e. dp/dx
% using symmetric finite difference
%
% usage: [j,ip] = jaco(fun,x0,V,verbose,order)
%
% (order 1:) Compute the 1st order partial derivatives (gradient) 
% of a function using:
%
% j(ip,:) = ( f(x(ip)+h)  - f(x(ip)-h) )  / (2 * h)
%
% (order 2:) Compute the 2nd order derivatives (curvature):
%
% j(ip,:) = [ (f0 - f1) / 2 / d ] ./ [ (f0 - 2 * fx + f1) / d ^ 2 ]
%
%
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

end

warning on;
if verbose
    fprintf('\n');
end

end