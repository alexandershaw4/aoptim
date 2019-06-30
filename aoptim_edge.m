function [X,F] = aoptim_edge(fun,x0,V,y,maxit,type)
global aopt
% minimise a problem of the form:
%
% y = f(x)
% e = sum(Y0 - y).^2
%
% usage:
%   [X,F] = aoptim_edge(fun,x0,V,y,maxit,type)
%
% fun = functional handle / anonymous function
% x0  = starting points (vector input to fun)
% V   = variances for each element of x0
% y   = Y0, for computing the objective: e = sum(Y0 - y).^2
% maxit = number of iterations (def=128)
% type = 'lin' or 'pol'  - locally linear model or polynomial (def lin)
%
% 
% To fit problems of the form:
% 
% e = f(x)
%
% usage - set y=0:
%   [X,F] = aoptim_edge(fun,x0,V,0,maxit,type)
%
%
% AS

if nargin < 6 || isempty(type)
    type = 'lin';
end

if nargin < 5 || isempty(maxit)
    maxit = 128;
end

% check functions
%--------------------------------------------------------------------------
aopt.fun  = fun;
aopt.y    = y(:);
x0        = full(x0(:));
V         = full(V(:));
[e0,er,Q] = obj(x0);

n         = 0;
iterate   = true;
criterion = 1e-2;
doplot    = 1;

V  = smooth(V);
Vb = V;

% initial grid reolution
gridn = 10000;
gridn = 100;

% initial point plot
if doplot
    makeplot(x0);
end

% print type of model
switch lower(type)
    case {'lin' 'linear'}; fprintf('Using locally linear model\n');
    case {'pol' 'poly' 'polinomial'}; fprintf('Using polynomial model\n');
end

n_reject_consec = 0;

% start loop
while iterate
    
    % counter
    n = n + 1;
   
    % construct an optimiser
    [de,dp,V] = pol_opt(x0,V,gridn,type);
    
    % ignore complex parameter values
    dp = real(dp);
    
    % assess output
    if de < e0
        x0    = dp;
        e0    = de;
        V     = V ./ 1.5;   % increase precision 
        %gridn = gridn * 10; % refine grid
        
        pupdate(n,de,e0,'accept');
        if doplot; makeplot(x0); end
        n_reject_consec = 0;
    else
        pupdate(n,de,e0,'reject');
        
        % reset grid and variance
        V               = V * 10;
        gridn           = gridn * 10;
        n_reject_consec = n_reject_consec + 1;
        
        % invoke a sampling routine temporarily?
        %[x0,e0]  = agradoptimi(fun,x0,V,y,1);
    end
        
    % stop at max iterations
    if n == maxit
        X = x0;
        F = e0;
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf('Convergence.\n');
        X = x0;
        F = e0;
        return;
    end
    
    % don't go endlessly without improvement
    if n_reject_consec >= 3 && n_reject_consec < 10
        % refine /reset the grid resolution
        gridn = 10000/2;
        % reset V
        V = Vb;        
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 10
        fprintf('Failed to converge... \nReturning best estimates.\n');
        X = x0;
        F = e0;
        return;
    end
    
end



end

function [e,x1,V] = pol_opt(x0,V,gridn,type)
global aopt

x0        = full(x0(:));
V         = full(V(:));
[e0,er,Q] = obj(x0);

% sample the edges of each parameter first - define a linear model
%--------------------------------------------------------------------------
LB = x0 - (x0.*V);
UB = x0 + (x0.*V);

for i   = 1:length(x0)
    
    % print loop update
    if i > 1; fprintf(repmat('\b',[1 length(str)] )); end
    str = sprintf('Parameter: %d/%d',i,length(x0));
    fprintf(str);
    
    % test upper bounds
    Uv    = x0;
    Uv(i) = UB(i); 
    [Ue(i),jeU(i,:)] = obj(Uv);
    
    % test lower bounds
    Lv    = x0;
    Lv(i) = LB(i);
    [Le(i),jeL(i,:)] = obj(Lv);
    
    % quarter bounds (upper)
    Uv    = x0;
    Uv(i) = mean([UB(i),x0(i)]);
    [Ue1(i),jeU1(i,:)] = obj(Uv);
    
    % quarter bounds (lower)
    Lv    = x0;
    Lv(i) = mean([LB(i),x0(i)]);
    [Le1(i),jeL1(i,:)] = obj(Lv);
end


% make an n-dim linear or poly model, passing through fp: f(0)
%--------------------------------------------------------------------------
for i = 1:length(x0)
    px     = [ LB(i) mean([LB(i),x0(i)]) x0(i) mean([UB(i),x0(i)]) UB(i) ];
    ex     = [ Le(i) Le1(i)              e0    Ue1(i)              Ue(i) ]; 
    
    chunks = gridn/length(px);
    dpx    = [];
    dex    = [];
    
    switch lower(type)
        
        case {'lin' 'linear'}
    
            for j  = 1:length(px) - 1
                dpx = [dpx linspace(px(j),px(j+1),chunks) ];
                dex = [dex linspace(ex(j),ex(j+1),chunks) ];
            end
            px = dpx;
            ex = dex;

            [~,I]  = min(ex);
            dP(i)  = px(I);
            Ers(i) = ex(I);
            
        case {'pol' 'poly' 'polinomial'}
            
            dex = linspace(ex(1),ex(end),chunks);
            p0  = polyfit(ex,px,3);
            dpx = polyval(p0, dex);
                
            px  = dpx;
            ex  = dex;

            [~,I]  = min(ex);
            dP(i)  = px(I);
            Ers(i) = ex(I);
    
    end
end

BadFit = find(isnan(dP));
dP(BadFit) = x0(BadFit);


% for i = 1:length(x0)
%     px     = [ LB(i) x0(i) UB(i) ];
%     ex     = [ Le(i) e0    Ue(i) ]; 
%     
%     px    = [linspace(px(1),px(2),gridn/2) linspace(px(2),px(3),gridn/2)];
%     ex    = [linspace(ex(1),ex(2),gridn/2) linspace(ex(2),ex(3),gridn/2)];
%     
%     [~,I]  = min(ex);
%     dP(i)  = px(I);
%     Ers(i) = ex(I);
% end

%[~,Ordr] = maxpoints(Ers,length(Ers));

% construct new p vector & error
%--------------------------------------------------------------------------
x1       = dP';
x1       = x1;%./V;   % new parameter estimates
[e,er,Q] = obj(x1);



% % make linear model, passing through fp: f(0)
% for i = 1:length(x0)
%     px     = [ LB(i) x0(i) UB(i) ];
%     ex     = [ Le(i) e0    Ue(i) ];   
%     B(i,:) = polyfit(ex,px,1);
% end
% 
% % compute some predicted parameters for e=0
% ip = find(B(:,1)~=0);
% for i = 1:length(ip)
%     ex    = [ Le(ip(i)) e0    Ue(ip(i)) ];  
%     dP(i) = polyval(B(ip(i),:),0);
% end
% 
% % test new parameter predictions
% x1       = x0;
% x1(ip)   = dP;
% [e,er,Q] = obj(x1);

% % convert edge-sampling to jacobian matrix
% d   = repmat( UB - LB , [1 size(jeU,2)] );
% Jac = (jeU - jeL) ./ ( 2*d );
% cj  = ( V'*cov(Jac') )';
% 
% % update non-zero variance terms
% iv     = find(cj); 
% V(iv)  = cj(iv);

end

function pupdate(it,err,best,action)

fprintf('| Main It: %04i | Err: %04i | Best: %04i | %s |\n',it,err,best,action);

end

function makeplot(x)

% compute objective and get data and model preidction
[e,er,Q,Y,y] = obj(x);

if iscell(Y)
    plot(spm_cat(Y),':'); hold on;
    plot(spm_cat(y)    ); hold off;
    drawnow;
end


end

function [e,er,Q,Y,y] = obj(x0)
global aopt

IS = aopt.fun;
P  = x0(:);

y  = IS(P); 
Y  = aopt.y;
Q  = 1;

try;   e  = sum( Q*(spm_vec(Y) - spm_vec(y)).^2 );
catch; e  = sum(   (spm_vec(Y) - spm_vec(y)).^2 );
end

% error along output vector
er = spm_vec(y) - spm_vec(Y);

end