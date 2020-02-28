function [X,F] = AOsampler(fun,x0,V,y,maxit,inner_loop,Q,criterion,min_df,mimo,order,writelog,objective)
% Gradient/curvature descent based optimisation, primarily for model fitting
% [system identification & parameter estimation]
%
% Fit multivariate linear/nonlinear models of the forms:
%   Y0 = f(x) + e   ..or
%   e  = f(x)
%
% Y0 = empirical data (to fit)
% f  = model (function)
% x  = model parameters/inputs to be optimised
%
% The output of fun(x) can be either a single value or a vector. The
% derivatives can be returned w.r.t the objective (SSE, MSE, FE etc, e.g. a MISO), or 
% w.r.t the error at each point along the output (e.g. a MIMO). To invoke 
% the latter, use the 10th input [0/1] to flag.
%
% Usage 1: to minimise a model fitting problem of the form:
%--------------------------------------------------------------------------
%   y    = f(x)
%
%   e    = sum(Y0 - y).^2            ... (SSE) or
%   F(p) = log evidence - divergence ... (Free Energy)
%
% the usage is:
%   [X,F] = AO(fun,x0,V,y,maxit,inner_loop,Q,crit,min_df,mimo,ordr,writelog,obj)
%
% minimum usage (using defaults):
%   [X,F] = AO(fun,x0,V,[y])
%
% INPUTS:
% fun        = function handle / anonymous function
% x0         = starting points (vector input to fun)
% V          = variances controlling each element of x0
% y          = Y0 / the data to fit, for computing the objective: e = sum(Y0 - y).^2
% maxit      = number of iterations (def=128) to restart descent
% inner_loop = num iters to continue on a specific descent (def=9999)
% Q          = optional precision matrix of size == (fun(x),fun(x))
% crit       = convergence value @(e=crit)
% min_df     = minimum change in function value (Error) to continue
%              (set to -1 to switch off)
% mimo       = flag for a MIMO system: i.e. Y-fun(x) returns an error VECTOR
% order      = [-1, 0, 1, 2, 3, 4, 5] - ** see jaco.m for opts **
% writelog   = flag to write progress to command window (0) or a txt log (1)
% obj        = 'sse' 'free_energy' 'mse' or 'rmse'
%
% Notes: in my testing minimising SSE seems to produce the best overall fits but
% free energy (log evidence - divergence) converges much faster. 
%
% Usage 2: to minimise objective problems of the form:
%--------------------------------------------------------------------------
%   e = f(x)
%
% the usage is: [note: set y=0 if f(x) returns the error/objective to be minimised]
%   [X,F] = AO(fun,x0,V,0,maxit,inner_loop, ... ) 
%
%
% Example: Minimise Ackley function:
%--------------------------------------------------------------------------
%     arg min: e = 20*(1 - exp(-0.2*sqrt(0.5*(x1.^2 + x2.^2)))) ...
%                                - exp(0.5*(cos(2*pi*x1)...
%                                         + cos(2*pi*x2))) + exp(1);
%
%     [X,F] = AO(@ackley_fun,[3 .5],[1 1]/128,0,[],[],[],1e-13,0,0,2)
%
% See also ao_glm AO_DCM jaco AO
%
% AS2019
% alexandershaw4@gmail.com
%
global aopt

if nargin < 13 || isempty(objective);  objective = 'sse'; end
if nargin < 12 || isempty(writelog);   writelog = 0;      end   
if nargin < 11 || isempty(order);      order = 2;         end
if nargin < 10 || isempty(mimo);       mimo = 0;          end
if nargin < 9  || isempty(min_df);     min_df = 0;        end
if nargin < 8  || isempty(criterion);  criterion = 1e-2;  end
if nargin < 7  || isempty(Q);          Q = 1;             end
if nargin < 6  || isempty(inner_loop); inner_loop = 9999; end
if nargin < 5  || isempty(maxit);      maxit = 128;       end
if nargin < 4  || isempty(y);          y = 0;             end

% Set up log if requested
persistent loc ;
if writelog
    name = datestr(now); name(name==' ') = '_';
    name = [char(fun) '_' name '.txt'];
    loc  = fopen(name,'w');
else
    loc = 1;
end

% check functions, inputs, options...
%--------------------------------------------------------------------------
%aopt         = [];       % reset
aopt.order   = order;    % first or second order derivatives [-1,0,1,2]
aopt.fun     = fun;      % (objective?) function handle
aopt.y       = y(:);     % truth / data to fit
aopt.pp      = x0(:);    % starting parameters
aopt.Q       = Q;        % precision matrix: e = Q*(ey*ey')
aopt.history = [];       % error history when y=e & arg min y = f(x)
aopt.mimo    = mimo;     % flag compute derivs w.r.t multi-outputs
aopt.memory  = 0;        % incorporate previous gradients when recomputing
aopt.fixedstepderiv = 1; % fixed or adjusted step for derivative calculation
aopt.ObjectiveMethod = objective; % 'sse' 'fe' 'mse' 'rmse' (def sse)

% % if no prior guess for parameters step sizes (variances) find step sizes
% % whereby each parameter has similar effect size w.r.t. error
% if nargin < 3 || isempty(V)
%     % iterates: v = v./abs(J)
%     V = FindOptimumStep(x0,V);
% end

% parameter and step vectors
x0  = full(x0(:));
V   = full(V(:));
v   = V;
pC  = diag(V);

% variance (in reduced space)
%--------------------------------------------------------------------------
%V     = spm_svd(pC);
V     = eye(length(x0));    %turn off svd 
pC    = V'*pC*V;
ipC   = inv(spm_cat(spm_diag({pC})));
red   = diag(pC);

aopt.pC  = V*red;      % store for derivative & objective function access
aopt.ipC = ipC; 

% initial objective value
[e0]       = obj(x0);
n          = 0;
iterate    = true;
doplot     = 1;
Vb         = V;

% initial error plot(s)
%--------------------------------------------------------------------------
if doplot
    setfig(); makeplot(x0,x0);
end

% initialise counters
%--------------------------------------------------------------------------
n_reject_consec = 0;
search          = 0;

% parameters (in reduced space)
%--------------------------------------------------------------------------
np    = size(V,2); 
p     = [V'*x0];
ip    = (1:np)';
Ep    = V*p(ip);

dff          = []; % tracks changes in error over iterations
localminflag = 0;  % triggers when stuck in local minima

% print start point - to console or logbook (loc)
refdate(loc);
pupdate(loc,n,0,e0,e0,'start:');

% start loop
%==========================================================================
while iterate
    
    % counter
    %----------------------------------------------------------------------
    n = n + 1;    tic;
   
    pupdate(loc,n,0,e0,e0,'sample',toc);
    
    aopt.pp = x0;
    
    % compute gradients & search directions
    %----------------------------------------------------------------------
    %[e0,df0] = obj( V*x0(ip) );
    retp = x0;
    rete = e0;
    
    rng default;
    for is = 1:4
        
        % Sample
        for i = 1:32
            qS     = spm_sqrtm(diag(red));
            P(:,i) = x0 + qS*randn(length(x0),1);
            R(:,i) = obj(P(:,i));
            
            if i > 1; fprintf(repmat('\b',[1 length(str)])); end
            str = sprintf('Sampling(%d): %d/%d ',is,i,32);
            fprintf(str);
        end
        fprintf('\n');
        
        % Assess
        [e1,j] = min(R);
        if e1  < e0
            
            % print / plot
            pupdate(loc,n,is,R(:,j),e0,'accept',toc);
            if doplot; makeplot(P(:,j),x0); end
        
            % Accept
            dx = P(:,j) - x0;
            de = P(:,j) - e0;
            x0 = P(:,j);
            e0 = R(:,j);
            
            % Retain Parameters and Error
            retp = [retp x0];
            rete = [rete e0];
            
            % Extrapolate
            ext = true;
            nex = 0;
            while ext
                nex = nex + 1;
                ddx = x0  + (dx/nex);
                dde = obj(ddx);
                
                if dde < e0
                    pupdate(loc,n,is,dde,e0,'extrap',toc);
                    x0 = ddx;
                    e0 = dde;
                else
                    ext = false;
                end
            end % end extrap while
            
        end % end if improve
        fprintf('\n');    
    end
    
    % Do something with the retained information: Estimate gradients?
    retp;
    
    % stopping criteria, rules etc.
    %======================================================================    
    % stop at max iterations
    if n == maxit
        fprintf(loc,'Reached max iterations: stopping\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        if writelog;fclose(loc);end
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf(loc,'Convergence.\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                
        if writelog;fclose(loc);end
        return;
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 5
        fprintf(loc,'Failed to converge... \nReturning best estimates.\n');
        
        % return current best
        X = V*real(x0(ip));
        F = e0;
                        
        if writelog;fclose(loc);end
        return;
    end
    
end



end

function refdate(loc)
fprintf(loc,'\n');

fprintf(loc,'| ITERATION     | FUN EVAL | CURRENT ERROR     | BEST ERROR SO FAR  | ACTION | TIME\n');
fprintf(loc,'|---------------|----------|-------------------|--------------------|--------|-------------\n');

end

function prinfo(loc,it,nfun,nc,ncs)

fprintf(loc,'| Main It: %04i | nf: %04i | Selecting components: %01i of %01i\n',it,nfun,nc,ncs);

end

function pupdate(loc,it,nfun,err,best,action,varargin)

if nargin >= 7
    n = varargin{1};
    fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s | %d\n',it,nfun,err,best,action,n);
else
    fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s |\n',it,nfun,err,best,action);
end

end

function setfig()

figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off','position',[1088 122 442 914]);
set(gcf, 'MenuBar', 'none');
set(gcf, 'ToolBar', 'none');
drawnow;
    
end

function makeplot(x,ox)
% plot the function output (f(x)) on top of the thing we're ditting (Y)
%
%
global aopt

[Y,y] = GetStates(x);

% Restrict plots to real values only - just for clarity really
% (this doesn't mean the actual model output is not complex)
Y = spm_unvec( real(spm_vec(Y)), Y);
y = spm_unvec( real(spm_vec(y)), y);

if ~isfield(aopt,'oerror')
    aopt.oerror = spm_vec(Y) - spm_vec(y);
end

former_error = aopt.oerror;
new_error    = spm_vec(Y) - spm_vec(y);

if length(y)==1 && length(Y) == 1 && isnumeric(y)
    ax        = gca;
    ax.YColor = [1 1 1];
    ax.XColor = [1 1 1];
    ax.Color  = [.3 .3 .3];hold on;
    
    % memory based error trace when y==e
    aopt.history = [aopt.history y];
    plot(aopt.history,'wo');hold on;
    plot(aopt.history,'w');hold off;grid on;
    ylabel('Error^2');xlabel('Step'); 
    title('AO: System Identification: Error','color','w','fontsize',18);
    drawnow;
elseif iscell(Y) && all( size(Y{1}) > 1)
    % Matrix representation (imagesc)
    py = spm_unvec( spm_vec(y), Y);
    subplot(321);imagesc(Y{1});
    subplot(322);imagesc(py{1});
    title('AO: System Identification','color','w','fontsize',18);
    
    s(2) = subplot(3,2,[3 4]);
    %bar([former_error new_error]);
    plot(former_error,'w--','linewidth',3); hold on;
    plot(new_error,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('Error Change','color','w','fontsize',18);
    s(2).YColor = [1 1 1];
    s(2).XColor = [1 1 1];
    s(2).Color  = [.3 .3 .3];
    
    
    s(3) = subplot(3,2,[5 6]);
    bar([ x(:)-ox(:) ],'FaceColor',[1 .7 .7],'EdgeColor','w');
    title('Parameter Change','color','w','fontsize',18);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s(3).YColor = [1 1 1];
    s(3).XColor = [1 1 1];
    s(3).Color  = [.3 .3 .3];
    drawnow;
    
else
%if iscell(Y)
    s(1) = subplot(311);
    plot(spm_cat(Y),'w:','linewidth',3); hold on;
    plot(spm_cat(y),     'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('AO: System Identification','color','w','fontsize',18);
    s(1).YColor = [1 1 1];
    s(1).XColor = [1 1 1];
    s(1).Color  = [.3 .3 .3];

    s(2) = subplot(312);
    %bar([former_error new_error]);
    plot(former_error,'w--','linewidth',3); hold on;
    plot(new_error,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('Error Change','color','w','fontsize',18);
    s(2).YColor = [1 1 1];
    s(2).XColor = [1 1 1];
    s(2).Color  = [.3 .3 .3];
    
    
    s(3) = subplot(313);
    bar([ x(:)-ox(:) ],'FaceColor',[1 .7 .7],'EdgeColor','w');
    title('Parameter Change','color','w','fontsize',18);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s(3).YColor = [1 1 1];
    s(3).XColor = [1 1 1];
    s(3).Color  = [.3 .3 .3];
    drawnow;
%end
end

aopt.oerror = new_error;

end

function [Y,y] = GetStates(x)
% - evaluates the model and returns it along with the stored data Y
%

global aopt

IS = aopt.fun;
P  = x(:)';

try    y  = IS(P); 
catch; y  = spm_vec(aopt.y)*0;
end
Y  = aopt.y;

end

function [e,J,er] = obj(x0)
% - compute the objective function - i.e. the sqaured error to minimise
% - also returns the parameter Jacobian
%

global aopt

method = aopt.ObjectiveMethod;

IS = aopt.fun;
P  = x0(:)';

try    y  = IS(P); 
catch; y  = spm_vec(aopt.y)*0+inf;
end

Y  = aopt.y;
Q  = aopt.Q;

%e  = sum( (spm_vec(Y) - spm_vec(y)).^2 );

if all(size(Q)>1) && ~strcmp(lower(method),'fe')
    
    % if square precision matrix was supplied, use this objective
    ey  = spm_vec(Y) - spm_vec(y);
    Q   = spm_unvec( rescale(spm_vec(Q),0,1), Q);
    eh  = (ey*ey').^2;
    qh  = full(Q+1).*eh;
    e   = sum(qh(:));
    %qh  = Q*(ey*ey') ;%*Q';    % i don't think the second Q term is needed
    %e   = sum(qh(:).^2);    
    e   = abs(e);
else
   
    if ~isfield(aopt,'J')
        aopt.J = ones(length(x0),length(spm_vec(y)));
    end
    if isfield(aopt,'J') && isvector(aopt.J)
        aopt.J = repmat(aopt.J,[1 length(spm_vec(y))]);
    end
    
    switch lower(method)
        case {'free_energy','fe','freeenergy','logevidence'};
    
            % Free Energy Objective Function: F(p) = log evidence - divergence
            %----------------------------------------------------------------------
            Q  = spm_Ce(1*ones(1,length(spm_vec(y))));
            h  = sparse(length(Q),1) - log(var(spm_vec(Y))) + 4;
            iS = sparse(0);

            for i  = 1:length(Q)
                iS = iS + Q{i}*(exp(-32) + exp(h(i)));
            end

            ny  = length(spm_vec(y));
            nq  = ny ./ length(Q);
            e   = spm_vec(Y) - spm_vec(y);
            ipC = aopt.ipC;
            warning off; % don't warn abour singularity
            Cp  = spm_inv( (aopt.J*iS*aopt.J') + ipC );
            warning on
            p   = ( x0(:) - aopt.pp(:) );

            if any(isnan(Cp(:))) 
                Cp = Cp;
            end
            
            L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2;            ...
            L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;
           %L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2; % no hyperparameters
            F    = sum(L);
            e    = (-F);
            
            aopt.Cp = Cp;
            %aopt.Q  = iS;
            
            if strcmp(lower(method),'logevidence')
                % for log evidence, ignore the parameter term
                % its actually still an SSE measure really
                F = L(1);
                e = -F;
            end
    
        % Other Objective Functions
        %------------------------------------------------------------------ 
        case 'sse'
            % sse: sum of error squared
            e  = sum( (spm_vec(Y) - spm_vec(y) ).^2 ); e = abs(e);
            
        case 'sse2' % sse robust to complex systems
            % or
            e  = sum(sum( ( spm_vec(Y)-spm_vec(y)').^2 ));
            e  = real(e) + imag(e);
    
        case 'mse'
            % mse: mean squared error
            e = (norm(spm_vec(Y)-spm_vec(y),2).^2)/numel(spm_vec(Y));

        case 'rmse'
            % rmse: root mean squaree error
            e = ( (norm(spm_vec(Y)-spm_vec(y),2).^2)/numel(spm_vec(Y)) ).^(1/2);

        case {'correlation','corr','cor','r2'}
            % 1 - r^2 (bc. minimisation routine == maximisation)
            e = 1 - ( corr( spm_vec(Y), spm_vec(y) ).^2 );

        case 'combination'
            % combination:
            SSE = sum( ((spm_vec(Y) - spm_vec(y)).^2)./sum(spm_vec(Y)) );
            R2  = 1 - abs( corr( spm_vec(Y), spm_vec(y) ).^2 );
            e   = SSE + R2;

            % complex output models
            %ey  = spm_vec(Y) - spm_vec(y);
            %qh  = real(ey)*real(ey') + imag(ey)*imag(ey');
            %e   = sum(qh(:).^2);
    end
end

% error along output vector
er = spm_vec(y) - spm_vec(Y);

% this wraps obj to return only the third output for MIMOs
% when we want the derivatives w.r.t. each output 
function er  = inter(x0)
    [~,~,er] = obj(x0);
end

J = [];

% this hands off to jaco, which computes the derivatives
% - note this can be used differently for multi-output systems
if nargout == 2
    V    = aopt.pC;
    Ord  = aopt.order; 
    mimo = aopt.mimo;
    
    if aopt.fixedstepderiv == 1
        V = (~~V)*1e-3;
        %V = (~~V)*1e-4;
    end
    
    if ~mimo; [J,ip] = jaco(@obj,x0,V,0,Ord);    ... df[e]   /dx [MISO]
    else;     [J,ip] = jaco(@inter,x0,V,0,Ord);  ... df[e(k)]/dx [MIMO]
    end

    % store for objective function
    aopt.J = J;
    
    % accumulate gradients / memory of gradients
    if aopt.memory
        try
            J       = J + (aopt.pJ/2) ;
            aopt.pJ = J;
        catch
            aopt.pJ = J;
        end
    end
    
    %[J,ip] = jaco(IS,P',V,0,Ord);   ... df/dx
    %J = repmat(V,[1 size(J,2)])./J;
end


end

function V = FindOptimumStep(x0,v)
global aopt

fprintf('Auto computing parameter step sizes...(wait)\n');tic;

% initialise at 1/8 - arbitrary
% this is equivalent to:
% dx[i] = x[i] + ( x[i]*1/8 )
%
% using the n-th order numerical derivatives as a measure of parameter
% effect size, find starting step sizes whereby all parameters are
% effective / equally balanced
if nargin < 2 || isempty(v)
    v = ones( size(x0) )/8;
end
n = 0;

aopt.ipC   = inv(spm_cat(spm_diag({diag(v)})));

[J,ip] = jaco(@obj,x0,v,0,aopt.order);
n = 0;

while var(J) > 0.15
    vj = v./abs(J);
    vj(isinf(vj))=0;
    vj(isnan(vj))=0;
    vj = full(vj);
    v  = pinv( vj )' ;
    [J,ip] = jaco(@obj,x0,v,0,aopt.order);
    n = n + 1;
end

V = v;

fprintf('Finished computing step sizes in %d iterations (%d s)\n',n,round(toc));

end