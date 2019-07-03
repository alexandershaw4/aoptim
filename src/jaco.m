function [j,ip] = jaco(fun,x0,V,verbose)
% Compute the Jacobian matrix (parameter gradients) for a model using:
%
% j(ip,:) = ( f(x(ip)+h)  - f(x(ip)-h) )  / (2 * h)
%
%
% AS2019

if nargin < 4 || isempty(verbose)
    verbose = 0;
end

IS = fun;
P  = x0(:);

% if nargin == 3; ip = find(V(:));
% else;           ip = 1:length(x0);
% end

if nargin == 3; ip = ~~(V(:));
else;           ip = 1:length(x0);
end

j  = jacf(IS,P,ip,verbose);

j(isnan(j)) = 0;

end



function j = jacf(IS,P,ip,verbose)

% Compute the Jacobian matrix using variable step-size
n  = 0;
warning off ;

%f0    = feval(IS,P);
f0    = spm_cat( feval(IS,P) );
j     = zeros(length(P),length(f0(:))); % n param x n output
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
        d      = P0(i) * 0.01;
        
        if d == 0
            d = 0.01;
        end
        
        P0(i)  = P0(i) + d  ;
        P1(i)  = P1(i) - d  ;
                
        f0     = spm_vec(spm_cat(feval(IS,P0)));
        f1     = spm_vec(spm_cat(feval(IS,P1)));
        j(i,:) = (f0 - f1) / (2 * d);
        
        % Alternatively, include curvature
        %deriv1 = (f0 - f1) / 2 / d;
        %deriv2 = (f0 - 2 * fx + f1) / d ^ 2;
        %j(i,:) = deriv1 ./ deriv2;
    end
end

warning on;
if verbose
    fprintf('\n');
end

end