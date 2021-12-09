function [X,F,Cp,PP,Hist] = AO(funopts)
% A gradient/curvature descent optimisation routine, designed primarily 
% for nonlinear model fitting / system identification & parameter estimation. 
% 
% The objective function minimises the free energy (~ ELBO - evidence lower bound).
%
% Y0 = f(x) + e  
%
% Y0 = empirical data (to fit)
% f  = model (function)
% x  = model parameters/inputs to be optimised 
%      (treated as Gaussians with variance V)
%
% Given the model function (f), parameters (x) and data (y), AO computes the
% objective function, F:
%
%   F(x) = log evidence(e,y) - divergence(x) 
%
% For a multivariate function f(x) where x = [p1 .. pn]' the ascent scheme
% is:
%
%   x[p,t+1] = x[p,t] + a[p] *-dFdx[p]
%
% Note the use of different step sizes, a[p], for each parameter.
% Optionally, low dimensional hyperparameter tuning can be used to find 
% the best step size a[p], by setting step_method = 6;
%
%   x[p,t+1] = x[p,t] + (b*V) *-dFdx[p]  ... where b is optimised using fminsearch
%
% where V maps between a (reduced) parameter subspace and the full vector.
% dFdx[p] are the partial derivatives of F, w.r.t parameters, p. (Note F = 
% the objective function and not 'f' - your function). See jaco.m for options, 
% although by default these are computed using a finite difference 
% approximation of the curvature, which retains the sign of the gradient:
%
% f0 = F(x[p]+h) 
% fx = F(x[p]  )
% f1 = F(x[p]-h) 
%
% j(p,:) =        (f0 - f1) / 2h  
%          ----------------------------
%            (f0 - 2 * fx + f1) / h^2  
%
% nb. - depending on the specified step method, a[] is computed from j & V
% using variations on:      (where V is the variance of each param)
% 
%  J      = -j ;
%  dFdpp  = approx -(J'*J); -- however, we normalise out the full magnitude 
%                              of the gradient using a LDL decomposition,
%                              such that;
%  dFdpp  = -L*(D./sum(D))*L' 
%  a      = (V)./(1-dFdpp);   
% 
% If step_method = 3, dFdpp is a small number and this reduces to a simple
% scale on the stepsize - i.e. a = V./scale. However, if step_method = 1, J
% is transposed, such that dFdpp is np*np. This leads to much bigger
% parameter steps & can be useful when your start positions are a long way
% from the solution. In either case, note that the variance term V[p] on the
% parameter distribution is a scaled version of the step size. 
% 
% Alternatively, step_method=6 invokes hyperparameter tuning of the step
% size.
%
% By default momentum is included (opts.im=1). The idea is that we can 
% have more confidence in parameters that are repeatedly updated in the 
% same direction, so we can take bigger steps for those parameters as the
% optimisation progresses.
%
% For each iteration of the ascent:
% 
%   dx[p]  = x[p] + a[p]*-dFdx
%
% With the MLE setting, the probability of each predicted dx[p] coming 
% from x[p] is computed (Pp) and incorporated into a NL-WLS implementation of 
% MLE:
%           
%   b(s+1) = b(s) - (J'*Pp*J)^-1*J'*Pp*r(s)
%   dx     = x - (a*b)
%  
% The {free energy} objective function minimised is
%==========================================================================
%  L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2; % complexity minus accuracy of states
%  L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;                            % complexity minus accuracy of parameters
%  L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;                            % complexity minus accuracy of precision (hyperparameter)
%  F    = -sum(L);
%
% *Alternatively, set opts.objective to 'rmse' 'sse' 'euclidean' ... 
%
% INPUTS:
%-------------------------------------------------------------------------
% To call this function using an options structure:
%-------------------------------------------------------------------------
% opts = AO('options');     % get the options struct
% opts.fun = @myfun         % fill in what you want...
% opts.x0  = [0 1];         % start param value
% opts.V   = [1 1]/8;       % variances for each param
% opts.y   = [...];         % data to fit - .e.g y = f(p) + e
% opts.maxit       = 125;   % max num iterations 
% opts.step_method = 1;     % step method - 1 (big), 3 (small) and 4 (vanilla)**.
% opts.im    = 1;           % flag to include momentum of parameters
% opts.order = 2;           % partial derivate fun option (see jaco.m): 1 or 2
% opts.hyperparameters = 0; % flag to do a grad ascent on the precision (3rd term in FE)
% opts.criterion    = -300  % convergence threshold
% opts.mleselect    = 0;    % maximum likelihood param selection
% opts.objective    = 'fe'; % objective fun: 'free energy', 'logevidence','sse', ...
% opts.writelog     = 0;    % flag to write logbook instead of to console
% opts.Q   = [];            % square matrix precision operator ( size=length(y) )
% opts.inner_loop = 10;     % limit on number of repeats on same decent (between grad comp)
% opts.DoMLE = 0;           % flag to perform a sort of MLE via WLS
% opts.force_ls = 0;        % flag to force a line search every time
% opts.ismimo = 0;          % compute dFdx of a multi-output function, not just objective value
% opts.gradmemory = 0;      % remember previous gradients
% opts.doparallel = 0;      % compute gradients using parfor
% opts.fsd = 1;             % use fixed step for derivative computation
% opts.allow_worsen = 0;    % allow objective to get worse sometimes
% opts.doimagesc = 0;       % if data/model outputs are matrix (not vector), plot as such
% opts.EnforcePriorProb = 0;% force the parameter updates to strictly adhere to prior distribution
% opts.FS = []              % feat selection function: FS(y)
% opts.userplotfun = [];    % inject a user plot function into the main display
% opts.corrweight = 1;      % weight error term by correlation
%
% [X,F,Cp,PP,Hist] = AO(opts);       % call the optimser, passing the options struct
%
% OUTPUTS:
%-------------------------------------------------------------------------
% X   = posterior parameters
% F   = fit value (depending on objective function specified)
% CP  = parameter covariance
% Pp  = posterior probabilites
% H   = history
%
% *If the optimiser isn't working well, try making V smaller!
%
% References
%-------------------------------------------------------------------------
%
% "Computing the objective function in DCM" Stephan, Friston & Penny
% https://www.fil.ion.ucl.ac.uk/spm/doc/papers/stephan_DCM_ObjFcn_tr05.pdf
%
% "The free energy principal: a rough guide to the brain?" Friston
% https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf
%
% "Likelihood and Bayesian Inference And Computation" Gelman & Hill 
% http://www.stat.columbia.edu/~gelman/arm/chap18.pdf
%
% For the nonlinear least squares MLE / Gauss Newton:
% https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
%
% For an explanation of momentum in gradient methods:
% https://distill.pub/2017/momentum/
%
% Approximation of derivaives by finite difference methods:
% https://www.ljll.math.upmc.fr/frey/cours/UdC/ma691/ma691_ch6.pdf
%
% For an explanation of normalised gradients in gradient descent
% https://jermwatt.github.io/machine_learning_refined/notes/3_First_order_methods/3_9_Normalized.html
%
% AS2019/2020/2021
% alexandershaw4@gmail.com

% Print the description of steps and exit
%--------------------------------------------------------------------------
if nargin == 1 && strcmp(lower(funopts),'help')
    PrintHelp(); return;
end
if nargin == 1 && strcmp(lower(funopts),'options');
    X = DefOpts; return;
end



% Inputs & Defaults...
%--------------------------------------------------------------------------
if isstruct(funopts)
   parseinputstruct(funopts);
else
   fprintf('You have to supply a funopts input struct now...\nTry AO(''options'')\n');
   return;
end

% Set up log if requested
persistent loc ;
if writelog
    name = datestr(now); name(name==' ') = '_';
    name = [char(fun) '_' name '.txt'];
    loc  = fopen(name,'w');
else
    loc = 1;
end

% If a feature selection function was passed, append it to the user fun
if ~isempty(FS) && isa(FS,'function_handle')
    params.FS = FS;
end

% check functions, inputs, options... note: many of these are set/returned
% by the subfunctions parseinputstruct and DefOpts()
%--------------------------------------------------------------------------
%aopt         = [];      % reset
aopt.x0x0    = x0;
aopt.order   = order;    % first or second order derivatives [-1,0,1,2]
aopt.fun     = fun;      % (objective?) function handle
aopt.yshape  = y;
aopt.y       = y(:);     % truth / data to fit
aopt.pp      = x0(:);    % starting parameters
aopt.Q       = Q;        % precision matrix
aopt.history = [];       % error history when y=e & arg min y = f(x)
aopt.memory  = gradmemory;% incorporate previous gradients when recomputing

aopt.fixedstepderiv  = fsd;% fixed or adjusted step for derivative calculation
aopt.ObjectiveMethod = objective; % 'sse' 'fe' 'mse' 'rmse' (def sse)
aopt.hyperparameters = hyperparams;
aopt.forcels         = force_ls;  % force line search
aopt.mimo            = ismimo;    % derivatives w.r.t multiple output fun
aopt.parallel        = doparallel; % compute dpdy in parfor
aopt.doimagesc       = doimagesc;  % change plot to a surface 
aopt.rankappropriate = rankappropriate; % ensure facorised rank aprop cov
aopt.do_ssa          = ssa;
aopt.corrweight      = corrweight;
aopt.factorise_gradients = factorise_gradients;

BayesAdjust = mleselect; % Select params to update based in probability
IncMomentum = im;        % Observe and use momentum data            
givetol     = allow_worsen; % Allow bad updates within a tolerance
EnforcePriorProb = EnforcePriorProb; % Force updates to comply with prior distribution
WeightByProbability = WeightByProbability; % weight parameter updates by probability
ExLineSearch = ext_linesearch;

params.aopt = aopt;      % Need to move form global aopt to a structure
params.userplotfun = userplotfun;

% parameter and step / distribution vectors
x0  = full(x0(:));
XX0 = x0;
V   = full(V(:));
v   = V;
pC  = diag(V);

crit = [0 0 0 0 0 0 0 0];

% variance (in reduced space)
%--------------------------------------------------------------------------
V     = eye(length(x0));    %turn off svd 
pC    = V'*(pC)*V;
ipC   = spm_inv(spm_cat(spm_diag({pC})));
red   = (diag(pC));

aopt.updateh = true; % update hyperpriors
aopt.pC  = red;      % store for derivative & objective function access
aopt.ipC = ipC;      % store ^

params.aopt = aopt;

% initial objective value (approx at this point as missing covariance data)
[e0]       = obj(x0,params);
n          = 0;
iterate    = true;
Vb         = V;
    
% initial error plot(s)
%--------------------------------------------------------------------------
if doplot
    setfig(); params = makeplot(x0,x0,params); aopt.oerror = params.aopt.oerror;
    pl_init(x0,params)
end

% initialise counters
%--------------------------------------------------------------------------
n_reject_consec = 0;
search          = 0;

% Initial probability threshold for inclusion (i.e. update) of a parameter
Initial_JPDtol = 1e-10;
JPDtol = Initial_JPDtol;

% parameters (in reduced space)
%--------------------------------------------------------------------------
np    = size(V,2); 
p     = [V'*x0];
ip    = (1:np)';
Ep    = V*p(ip);

dff          = []; % tracks changes in error over iterations
localminflag = 0;  % triggers when stuck in local minima

% print options before we start printing updates
if DoMLE;       fprintf('Using GaussNewton/MLE option\n'); end
if BayesAdjust; fprintf('Using Probability-Based Parameter Selection option\n'); end
if IncMomentum; fprintf('Using Momentum option\n');    end
fprintf('Using step-method: %d\n',step_method);
fprintf('Using Jaco (gradient) option: %d\n',order);
fprintf('User fun has %d varying parameters\n',length(find(red)));

% print start point - to console or logbook (loc)
refdate(loc);pupdate(loc,n,0,e0,e0,'start: ');

if step_method == 0
    % step method can switch between 1 (big) and 3 (small) automatcically
     autostep = 1;
else autostep = 0;
end

% start optimisation loop
%==========================================================================
while iterate
    
    % counter
    %----------------------------------------------------------------------
    n = n + 1;    tic;
   
    pupdate(loc,n,0,e0,e0,'gradnts',toc);
    
    XX0 = x0;
        
    % compute gradients & search directions
    %----------------------------------------------------------------------
    aopt.updatej = true; aopt.updateh = true; params.aopt  = aopt;
    
    % Second order partial derivates of F w.r.t x0 using Jaco.m
    [e0,df0,~,~,~,~,params]  = obj( V*x0(ip),params );
    [e0,~,er] = obj( V*x0(ip),params );
    df0 = real(df0);
       
    % Update aopt structure and place in params
    aopt         = params.aopt;
    aopt.er      = er;
    aopt.updateh = false;
    params.aopt  = aopt;
                     
    % print end of gradient computation (just so we know it's finished)
    pupdate(loc,n,0,e0,e0,'grd-fin',toc); 
    
    % Switching for different methods for calculating 'a' / step size
    if autostep; search_method = autostepswitch(n,e0,Hist);
    else;        search_method = step_method;
    end
    
    % initial search direction (steepest) and slope
    %----------------------------------------------------------------------    
    % Compute step, a, in scheme: dx = x0 + a*-J
    if     n == 1 && search_method~=100 ; a = red*0; 
    elseif n == 1 && search_method==100 ; a = repmat({red*0},1,4); 
    end
    
    % Setting step size to 6 invokes low-dimensional hyperparameter tuning
    if search_method ~= 6
        pupdate(loc,n,0,e0,e0,'stepsiz',toc);
    else
        pupdate(loc,n,0,e0,e0,'hyprprm',toc);
    end
    
    % Normal search method
    if search_method ~= 100
        if ~ismimo
            [a,J] = compute_step(df0,red,e0,search_method,params,x0,a,df0); 
        else
            [a,J] = compute_step(params.aopt.J,red,e0,search_method,params,x0,a,df0); 
            J = -df0(:);
        end
    else
        % Greedy search method: force it to try a few different step 
        % computations and pick the best
        steps   = [1 3 4 7]; 
        for ist = 1:length(steps)
            [a{ist},J{ist}] = compute_step(df0,red,e0,steps(ist),params,x0,a{ist},df0);
        end
    end
     
    if search_method ~= 6
        pupdate(loc,n,0,e0,e0,'stp-fin',toc);
    else
        pupdate(loc,n,0,e0,e0,'hyp-fin',toc);
    end    
    
    % Log start of iteration (these are returned)
    Hist.e(n) = e0;
    Hist.p{n} = x0;
    Hist.J{n} = df0;
    Hist.a{n} = a;
    
    % Make copies of error and param set for inner while loops
    x1  = x0;
    e1  = e0;
        
    % Start counters
    improve = true;
    nfun    = 0;
    
    % iterative descent on this (-gradient) trajectory
    %======================================================================
    while improve
                
        % Log number of function calls on this iteration
        nfun = nfun + 1;
                                        
        % Compute The Parameter Step (from gradients and step sizes):
        % % x[p,t+1] = x[p,t] + a[p]*-dfdx[p] 
        %------------------------------------------------------------------if
        if ~ExLineSearch
            if search_method ~= 100
                dx   = compute_dx(x1,a,J,red,search_method,params);  % dx = x1 + ( a * J );  
                ddxm = dx;
            else
                % When we are doing a greedy search there are multiple dx's
                % resulting from different step size methods - try them all.
                aopt.updatej = false; % switch off objective fun triggers
                aopt.updateh = false;
                params.aopt  = aopt;
                for ist = 1:length(steps)
                    dx(:,ist) = compute_dx(x1,a{ist},J{ist},red,steps(ist),params);
                    ex(:,ist) = obj(dx(:,ist),params);
                end
                ddxm = dx;
            end
        else
            %fobj = @(x) obj_J(x,params);
            %[dx,~] = linesearch(fobj,y,x1,4,1e-5,50);
            %fobj = @(x) obj(x,params);
            %lso = lineSearch(fobj,x1,(df0./norm(df0)),'none');
        end
        
        % The following options are like 'E-steps' or line search options 
        % - i.e. they estimate the missing variables (parameter indices & 
        % values) that should be optimised in this iteration
        %==================================================================
        
        % Compute the probabilities of each (predicted) new parameter
        % coming from the same distribution defined by the prior (last best)
        
        % Loop if there are multiple steps being tested (i.e. ddxm is a matrix),
        % although this can also be because ismimo=1 (if dFdx was computed
        % on vector output), in which case stop it at 1
        if ~ismimo; nlp = size(ddxm,2);
        else;       nlp = 1;
        end
        
        for istepm = 1:nlp
                        
            % dx prediction from step method n
            dx = ddxm(:,istepm); 
        
            pupdate(loc,n,nfun,e1,e1,'p dists',toc);
            
            % Distributions
            pt  = zeros(1,length(x1));
            for i = 1:length(x1)
                %vv     = real(sqrt( red(i) ));
                vv     = real(sqrt( red(i) ))*2;
                if vv <= 0 || isnan(vv) || isinf(vv); vv = 1/64; end
                pd(i)  = makedist('normal','mu', real(aopt.pp(i)),'sigma', vv);
            end

            % Curb parameter estimates trying to exceed their distirbution bounds
            if EnforcePriorProb
                odx = dx;
                nst = 1;
                for i = 1:length(x1)
                    if red(i)
                        if dx(i) < ( pd(i).mu - (nst*pd(i).sigma) )
                           dx(i) = pd(i).mu - (nst*pd(i).sigma);
                        elseif dx(i) > ( pd(i).mu + (nst*pd(i).sigma) )
                               dx(i) = pd(i).mu + (nst*pd(i).sigma);
                        end
                    end
                end
            end

            % Compute relative change in 'probability'^
            pdx = pt*0;
            for i = 1:length(x1)
                if red(i)
                    %vv     = real(sqrt( red(i) ));
                    vv     = real(sqrt( red(i) ))*2;
                    if vv <= 0 || isnan(vv) || isinf(vv); vv = 1/64; end
                     pd(i)  = makedist('normal','mu', real(aopt.pp(i)),'sigma', vv);
                     pdx(i) = (1./(1+exp(-pdf(pd(i),dx(i))))) ./ (1./(1+exp(-pdf(pd(i),aopt.pp(i)))));
                else
                end
            end    
            pt = pdx;
            prplot(pt);

            % If WeightByProbability is set, use p(dx) as a weight on dx
            if WeightByProbability
                x1 = x1 + ( pt(:).*(dx-x1) );
            end

            % Save for computing gradient ascent on probabilities
            p_hist(n,:) = pt;
            Hist.pt(:,n)  = pt;

            % Update plot: probabilities
            [~,oo]  = sort(pt(:),'descend');
            probplot(cumprod(pt(oo)),0.95,oo);

            % This is a variation on the Gauss-Newton algorithm - compute 
            % MLE via WLS - where the weights are the probabilities
            %------------------------------------------------------------------
            if DoMLE                     
                iter_mle = true;
                nmfun    = 0;
                clear b
                
                while iter_mle
                    pupdate(loc,n,nmfun,e1,e1,'MLE/WLS',toc);
                    nmfun = nmfun + 1;
                    nfun  = nfun  + 1;
                    
                    % parameter derivatives
                    if ~ismimo;  j  = J(:)*er';
                    else;        j = aopt.J;
                    end
                    
                    % weights and residuals
                    w  = pt;
                    r0 = spm_vec(y) - spm_vec(params.aopt.fun(spm_unvec(x1,aopt.x0x0))); 

                    if ~isempty(FS)
                        r0 = FS(r0);
                    end

                    db  = ( pinv(j'*diag(w)*j)*j'*diag(w) )'*r0;
                    
                    % b(s+1) = b(s) - (J'*J)^-1*J'*r(s)
                    try    b = b - db;
                    catch; b = db;
                    end
                    
                    % inclusion of a weight essentially makes this a Marquardt regularisation parameter
                    if isvector(a)
                        dxd = x1 - a.*b;
                    else
                        dxd = x1 - a*b; % recompose dx including step matrix (a)
                    end
                    
                    % update x
                    if obj(dxd,params) < obj(x1,params) && (nmfun < 3)
                        x1 = dxd;
                        x0 = x1;
                        e1 = obj(x1,params);
                        e0 = e1;
                    else
                        iter_mle = false;
                        dx = dxd;
                    end
                end
            end

            % (option) Momentum inclusion
            %------------------------------------------------------------------
            if n > 2 && IncMomentum
                pupdate(loc,n,nfun,e1,e1,'momentm',toc);
                % The idea here is that we can have more confidence in
                % parameters that are repeatedly updated in the same direction,
                % so we can take bigger steps for those parameters
                imom = sum( diff(full(spm_cat(Hist.p))')' > 0 ,2);
                dmom = sum( diff(full(spm_cat(Hist.p))')' < 0 ,2);

                timom = imom >= (2);
                tdmom = dmom >= (2);

                moments = (timom .* imom) + (tdmom .* dmom);

                if any(moments)
                   % parameter update
                   ddx = dx - x1;
                   dx  = dx + ( ddx .* (moments./n) );
                end
            end

            % Given (gradient) predictions, dx[i..n], optimise obj(dx) 
            % Either by:
            % (1) just update all parameters
            % (2) update all parameters whose updated value improves obj
            % (3) update only parameters whose probability exceeds a threshold
            %------------------------------------------------------------------
            aopt.updatej = false; % switch off objective fun triggers
            aopt.updateh = false;
            params.aopt  = aopt;
            
            pupdate(loc,n,nfun,e1,e1,'eval dx',toc);

            if obj(dx,params) < obj(x1,params) && ~BayesAdjust && ~aopt.forcels
                % Don't perform checks, assume all f(dx[i]) <= e1
                % i.e. full gradient prediction over parameters is good and we
                % don't want to be explicitly Bayesian about it ...
                gp  = ones(1,length(x0));
                gpi = 1:length(x0);
                de  = obj(V*dx,params);
                DFE = ones(1,length(x0))*de; 
            else
                % Assess each new parameter estimate (step) individually
                if (~faster) || nfun == 1 % Once per gradient computation?
                    if ~doparallel
                        for nip = 1:length(dx)
                            XX     = V*x0;
                            if red(nip)
                                XX(nip)  = dx(nip);
                                DFE(nip) = obj(XX,params);
                            else
                                DFE(nip) = e0;
                            end
                        end
                    else
                        % Works way faster in parfor, ...
                        parfor nip = 1:length(dx)
                            XX     = V*x0;
                            if red(nip)
                                XX(nip)  = dx(nip);
                                DFE(nip) = obj(XX,params); 
                            else
                                DFE(nip) = e0;
                            end
                        end
                    end                

                    DFE  = real(DFE(:));

                    % Identify improver-parameters            
                    gp  = double(DFE < e0); % e0
                    gpi = find(gp);
                end

                if ~BayesAdjust
                    % If the full gradient prediction over parameters did not
                    % improve, but the BayesAdjust option is not selected, then
                    % only update parameters showing improvements in objective func
                    ddx        = V*x0;
                    ddx(gpi)   = dx(gpi);
                    dx         = ddx;
                    de         = obj(dx,params);
                    
                else
                    % MSort of maximum likelihood - opimise p(dx) according to 
                    % initial conditions (priors; a,b) and error
                    % arg max: p(dx | a,b & e)
                    
                    alpha = 0.95;
                    thresh = 1 - alpha ;

                    if n>1
                        thresh = 1 - (alpha - (1-(mean(1 - (p_hist(end,:)-p_hist(end-1,:))))) );
                    end

                    pupdate(loc,n,nfun,e1,e1,'mleslct',toc);

                    PP = pt(:) ;

                    % parameter selection based on probability:
                    % sort h2l th probabilities and find jpdtol intersect
                    [~,o]  = sort(PP(:),'descend');

                    epar = e0 - DFE(o);
                    px   = cumprod(PP(o));

                    pI   = find( px > thresh );
                    [~,pnt] = min( epar(pI) );

                    selpar = o(1:pnt); % activate parameters

                    newp         = x1;
                    newp(selpar) = dx(selpar);
                    %if nfun>1 && all(newp(:)~=dx(:))
                        dx           = newp(:);                
                        de           = obj(V*newp,params);
                    %else
                    %    improve = false;
                    %end

                end            
            end
            
            fprintf('--> euc dist(dp) = %d\n',cdist(dx',x1'));

           % Save for this step method...
           ddxm(:,istepm) = dx; 

        end
                        
        % Tolerance on update error as function of iteration number
        % - this can be helpful in functions with lots of local minima
        if givetol; etol = e1 * ( ( 0.5./(n*2) )  );
        else;       etol = 0; 
        end
        
        if etol ~= 0
            inner_loop=2;
        end
        
        deltap = cdist(dx',x1');
        deltaptol = 1e-3;
        
        % if running multiple search (step) methods, now is the time to
        % evaluate and commit to one for the remainder of the iteration
        if search_method == 100
            for ist = 1:size(ddxm,2)
                dex(ist) = obj(ddxm(:,ist),params);
            end
            [~,Ith]=min(real(dex));
            dx = ddxm(:,Ith);
            fprintf('\nSelecting step method %d\n',steps(Ith));
        end
          
        % linear model prediction using history
        %
        if n > 1
            %b_p = cat(2,Hist.p{:});
            % Hist.e(n);
            % Hist.p{n};
            % Hist.J{n};
        end
        
        % Evaluation of the prediction(s)
        %------------------------------------------------------------------
        if de  < ( obj(x1,params) + abs(etol) ) && (deltap > deltaptol)
            
            % If the objective function has improved...
            if nfun == 1; pupdate(loc,n,nfun,de,e1,'improve',toc); end
            
            % update the error & the (reduced) parameter set
            %--------------------------------------------------------------
            df  = e1 - de;
            e1  = de;
            x1  = V'*dx;
        else
            % If it hasn't improved, flag to stop this loop...
            improve = false;            
        end
        
        % upper limit on the length of this loop (force recompute dfdx)
        if nfun >= inner_loop
            improve = false;
        end
    end  % end while improve... ends iter descent on this trajectory
      
    
    % ignore complex parameter values - for most functions, yes
    %----------------------------------------------------------------------
    x1 = (x1); % (put 'real' here)
        
    % evaluate - accept/reject - plot - adjust rate
    %======================================================================
    if e1 < e0 && ~aopt.forcels && (deltap > deltaptol)  % Improvement...
        
        % Compute deltas & accept new parameters and error
        %------------------------------------------------------------------
        df =  e1 - e0;
        dp =  x1 - x0;
        x0 =  dp + x0;
        e0 =  e1;
                
        % Extrapolate...
        %==================================================================        
        
        % we know what param-step caused what improvement, so try again...
        %------------------------------------------------------------------
        exploit = true;
        nexpl   = 0;
        pupdate(loc,n,nexpl,e1,e0,'descend',toc);
                
        while exploit
            % local linear extrapolation
            extrapx = V*(x1+(-dp));
            if obj(extrapx,params) < (e1+abs(etol))
                x1    = extrapx(:);
                e1    = obj(x1,params);
                nexpl = nexpl + 1;
            else
                % if this didn't work, just stop and move on to accepting
                % the best set from the while loop above
                exploit = false;
                pupdate(loc,n,nexpl,e1,e0,'finish ',toc);
            end
            
            % upper limit on the length of this loop: no don't do this
            if nexpl == (inner_loop)
                exploit = false;
            end
        end
        
        % Update best-so-far estimates
        e0 = e1;
        x0 = x1;
            
        % Print & plots success
        %------------------------------------------------------------------
        nupdate = [length(find(x0 - aopt.pp)) length(x0)];
        pupdate(loc,n,nfun,e1,e0,'accept ',toc,nupdate);      % print update
        if doplot; params = makeplot(V*x0(ip),aopt.pp,params);aopt.oerror = params.aopt.oerror; end   % update plots
        
        n_reject_consec = 0;              % monitors consec rejections
        JPDtol          = Initial_JPDtol; % resets prob threshold for update

    else
        
        % *If didn't improve: invoke much more selective parameter update
        %==================================================================
        pupdate(loc,n,nfun,e1,e0,'parslct',toc);    
        e_orig = e0;
        
        % select only parameters whose steps improved the objective
        %------------------------------------------------------------------
        thisgood = gp*0; % track whether any of selection get used
        if any(gp)
            
            % sort good params by (improvement amount) * (probability)
            %--------------------------------------------------------------
            % update p's causing biggest improvment in fe while maintaining highest P(p)
            [~,PO] = sort(rescale(-DFE(gpi)),'descend');
            
            % loop the good params in selected (PO) order
            %--------------------------------------------------------------
            improve1 = 1;
            nimp     = 0 ;
            
            while improve1
                thisgood = gp*0; % tracks which params are updated below
                nimp     = nimp + 1;
                % evaluate the 'good' parameters
                for i  = 1:length(gpi)
                    xnew             = V*x0;
                    xnew(gpi(PO(i))) = dx(gpi(PO(i)));
                    enew             = obj(xnew,params);
                    % accept new error and parameters and continue
                    if enew < (e0+abs(etol)) && nimp < round(inner_loop)
                        x0  = V'*(xnew);
                        df  = enew - e_orig;
                        e0  = enew;
                        thisgood(gpi(PO(i))) = 1;
                    end
                end
                
                % Ok, now assess whether any parameters were accepted from
                % this selective search....
                %----------------------------------------------------------
                if any(thisgood)

                    % Print & plot update
                    nupdate = [length(find(x0 - aopt.pp)) length(x0)];
                    pupdate(loc,n,nfun,e0,e0,'accept ',toc,nupdate);
                    if doplot; params = makeplot(V*x0,x1,params);aopt.oerror = params.aopt.oerror; end

                    % Reset rejection counter & JPDtol
                    n_reject_consec = 0;
                    JPDtol = Initial_JPDtol;

                else
                    % If we made it to here, then neither the full gradient
                    % predicted update, nore the selective search, managed
                    % to improve the objective. So:
                    pupdate(loc,n,nfun,e0,e0,'reject ',toc);
                    
                    % Reduce param steps (vars) and go back to main loop
                    red = red*.8;
                    
                    warning off;try df; catch df = 0; end;warning on;
                    
                    % Halt this while loop % Keep counting rejections
                    improve1 = 0;
                    n_reject_consec = n_reject_consec + 1;
                    
                end
                
                % update global store of V
                aopt.pC = V*red;
            end
        else
            % Don't panic; it's ok 
            
            % If we get here, then there were not gradient steps across any
            % parameters which improved the objective! So...
            pupdate(loc,n,nfun,e0,e0,'reject ',toc);
            
            % Our (prior) 'variances' are probably too small
            red = red*1.4; 
            
            % Update global store of V & keep counting rejections
            aopt.pC = V*red;
            n_reject_consec = n_reject_consec + 1;
            
            warning off; try df; catch df = 0; end; warning on;
            
            % Loosen inclusion threshold on param probability (temporarily)
            % i.e. specified variances aren't making sense.
            % (applies to Bayes option only)
            if BayesAdjust
                JPDtol = JPDtol * 1e-10;
                pupdate(loc,n,nfun,e0,e0,'TolAdj ',toc);
            end
        end
                            
    end
    
    % Stopping criteria, rules etc.
    %======================================================================
    crit = [ (abs(df) < 1e-6) crit(1:end - 1) ]; 
    clear df;
    
    if all(crit)
        localminflag = 3;            
    end
        
    if localminflag == 3
        fprintf(loc,'We''re either stuck or converged...stopping.\n');
        
        % Return current best
        X = V*(x0(ip));
        F = e0;
               
        % Use best covariance estimate
        if doparallel
            aopt = params.aopt;
            Cp = spm_inv( (J(:)*J(:)')*aopt.ipC );
        else
            Cp = aopt.Cp;
        end
        
        % Peform Bayesian Inference
        PP = BayesInf(x0,Ep,diag(red));
        
        if writelog;fclose(loc);end
        return;
    end
    
    % If 3 fails, reset the reduction term (based on the specified variance)
    if n_reject_consec == 3
        pupdate(loc,n,nfun,e1,e0,'resetv');
        red     = diag(pC);
        aopt.pC = V*red;
        a       = red;
    end
    
    % stop at max iterations
    if n == maxit
        fprintf(loc,'Reached maximum iterations: stopping.\n');
        
        % Return current best
        X = V*(x0(ip));
        F = e0;
               
        % Use best covariance estimate
        if doparallel
            aopt = params.aopt;
            Cp = spm_inv( (J(:)*J(:)')*aopt.ipC );
        else
            Cp = aopt.Cp;
        end
        
        % Peform Bayesian Inference
        PP = BayesInf(x0,Ep,diag(red));
                
        if writelog;fclose(loc);end
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf(loc,'Convergence.\n');
        
        % Return current best
        X = V*(x0(ip));
        F = e0;
               
        % Use best covariance estimate
        if doparallel
            aopt = params.aopt;
            Cp = spm_inv( (J(:)*J(:)')*aopt.ipC );
        else
            Cp = aopt.Cp;
        end
        
        % Peform Bayesian Inference
        PP = BayesInf(x0,Ep,diag(red));
        
        if writelog;fclose(loc);end
        return;
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 5
        fprintf(loc,'Failed to converge...\n');
        
            % Return current best
            X = V*(x0(ip));
            F = e0;

            % Use best covariance estimate
            if doparallel
                aopt = params.aopt;
                Cp = spm_inv( (J(:)*J(:)')*aopt.ipC );
            else
                Cp = aopt.Cp;
            end

            % Peform Bayesian Inference
            PP = BayesInf(x0,Ep,diag(red));

            if writelog;fclose(loc);end
            return;
     end
end
    
end

% Subfunctions: Plotting & printing updates to the console / log...
%==========================================================================

function refdate(loc)
fprintf(loc,'\n');

fprintf(loc,'| ITERATION     | FUN EVAL | CURRENT F         | BEST F SO FAR      | ACTION  | TIME\n');
fprintf(loc,'|---------------|----------|-------------------|--------------------|---------|-------------\n');

end

function s = prinfo(loc,it,nfun,nc,ncs)

s = sprintf(loc,'| Main It: %04i | nf: %04i | Selecting components: %01i of %01i\n',it,nfun,nc,ncs);
fprintf(loc,'| Main It: %04i | nf: %04i | Selecting components: %01i of %01i\n',it,nfun,nc,ncs);

end

function s = pupdate(loc,it,nfun,err,best,action,varargin)
persistent tx
if nargin >= 7
    if length(varargin)==2
        nupdate = varargin{2};
    else
        nupdate = [];
    end
    
    if isempty(nupdate)
        n = varargin{1};
        fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s | %d\n',it,nfun,err,best,action,n);
        st=sprintf('\n| PROGRESS SUMMARY:\n|-----------------------------\n| Main Iteration: %04i \n| Num FunEval: %04i \n| Curr F: %04i \n| Best F: %04i \n| Status: %s \n| Time: %d\n',it,nfun,err,best,action,n);
    else
        n = varargin{1};
        fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s | %d\n',it,nfun,err,best,action,n);
        fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | N-p Updated: %d / %d\n',it,nfun,err,nupdate(1),nupdate(2));
        st=sprintf('\n| PROGRESS SUMMARY:\n|-----------------------------\n| Main Iteration: %04i \n| Num FunEval: %04i \n| Curr F: %04i \n| Best F: %04i \n| Status: %s \n| Time: %d\n',it,nfun,err,best,action,n);
    end
else
    fprintf(loc,'| Main It: %04i | nf: %04i | Err: %04i | Best: %04i | %s |\n',it,nfun,err,best,action);
    st = sprintf('\n| PROGRESS SUMMARY:\n|-----------------------------\n| Main It: %04i \n| Num FunEval: %04i \n| Curr F: %04i \n| Best F: %04i \n| Status: %s \n\n',it,nfun,err,best,action);
end

s = subplot(4,3,3);
delete(tx);
tx = text(0,.5,st,'FontSize',18,'Color','w');
ax = gca;
set(ax,'visible','off');
ax.XGrid = 'off';
ax.YGrid = 'on';
s.YColor = [1 1 1];
s.XColor = [1 1 1];
s.Color  = [.3 .3 .3];
drawnow;


end

function search_method = autostepswitch(n,e0,Hist)
% Auto-switching routine to find a good method for computing the step size
% (a) - alternating between big steps (==1) and small, generic GD steps (==3)
if n < 3
    search_method = 1;
else
    % auto switch the step size (method) based on mean negative error growth rate
    if (e0 ./ Hist.e(n-1)) > .9*mean([Hist.e e0] ./ [Hist.e(2:end) e0 e0])
        search_method = 3;
    else
        search_method = 1;
    end
end
end
        

function setfig()
% Main figure initiation

%figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off','position',[1088 122 442 914]);
figpos = get(0,'defaultfigureposition').*[1 1 0 0] + [0 0 710 842];
figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off','position',figpos); % [2436,360,710,842]
set(gcf, 'MenuBar', 'none');
set(gcf, 'ToolBar', 'none');
drawnow;
set(0,'DefaultAxesTitleFontWeight','normal');
end

function BayesPlot(x,pr,po)
% Update the Bayes plot

s(3) = subplot(3,2,5);
imagesc(pr);
s(4) = subplot(3,2,6);
imagesc(po);

ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
s(3).YColor = [1 1 1];
s(3).XColor = [1 1 1];
s(3).Color  = [.3 .3 .3];
s(4).YColor = [1 1 1];
s(4).XColor = [1 1 1];
s(4).Color  = [.3 .3 .3];
drawnow;
end

function pl_init(x,params)
% A subplot in the main figure of the start point of the algorithm
[Y,y] = GetStates(x,params);

% Restrict plots to real values only - just for clarity really
% (this doesn't mean the actual model output is not complex)
Y = spm_unvec( real(spm_vec(Y)), Y);
y = spm_unvec( real(spm_vec(y)), Y);

s(1) = subplot(4,3,[10 11]);

plot(spm_vec(Y),'w:','linewidth',3); hold on;
plot(spm_vec(y),     'linewidth',3,'Color',[1 .7 .7]); hold off;
grid on;grid minor;title('Start Point','color','w','fontsize',18);
s(1).YColor = [1 1 1];
s(1).XColor = [1 1 1];
s(1).Color  = [.3 .3 .3];

end

function prplot(pt)
% Updating probability subplot

s = subplot(4,3,8);
    bar(real( pt(:) ),'FaceColor',[1 .7 .7],'EdgeColor','w');
    title('P(dp)','color','w','fontsize',18);
    ylabel('P(dp)');
    ylim([0 1]);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s.YColor = [1 1 1];
    s.XColor = [1 1 1];
    s.Color  = [.3 .3 .3];
    drawnow;


end

function probplot(growth,thresh,oo)
% Updating probability subplot

growth=growth(oo);
growth(isnan(growth))=0;
    these = find(growth>thresh);
    s = subplot(4,3,[9]);
    bar(real(growth),'FaceColor',[1 .7 .7],'EdgeColor','w');
    %plot(growth,'w','linewidth',3);hold on
    %plot(these,growth(these),'linewidth',3,'Color',[1 .7 .7]);
    title('P(p | prior N(μ,σ2)) ','color','w','fontsize',18);
    ylim([0 1]);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s.YColor = [1 1 1];
    s.XColor = [1 1 1];
    s.Color  = [.3 .3 .3];
    
    %plot(1:length(growth),growth*0+thresh)
    hold off;
    drawnow;
end

function params = makeplot(x,ox,params)
% Main updating plot the function output (f(x)) on top of the thing we're 
% fitting (Y)
%

aopt = params.aopt;

% User may pass a plot function which accepts parameter vector x and
% params structure, and calls the user function (the one being optimised):
% for example:
%
%
% function aodcmplotfun(p,params)
%    [~,~,y,t] = params.aopt.fun(p,params);
%    plot(t,y);
%
% note: this plot will be added to the main AO optimisation figure in
% position subplot(4,3,12) and coloured appropriately...
if isfield(params,'userplotfun') && ~isempty(params.userplotfun);
    subplot(4,3,12);
    feval(params.userplotfun,x,params);
    ax        = gca;
    ax.YColor = [1 1 1];
    ax.XColor = [1 1 1];
    ax.Color  = [.3 .3 .3];hold on;
end


[Y,y] = GetStates(x,params);

% Restrict plots to real values only - just for clarity really
% (this doesn't mean the actual model output is not complex)
Y = spm_unvec( real(spm_vec(Y)), Y);
y = spm_unvec( real(spm_vec(y)), Y);

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
    title('AO: Parameter Estimation: Error','color','w','fontsize',18);
    drawnow;
else
    if ~aopt.doimagesc
        s(1) = subplot(4,3,[1 2]);

        plot(spm_vec(Y),'w:','linewidth',3); hold on;
        plot(spm_vec(y),     'linewidth',3,'Color',[1 .7 .7]); hold off;
        grid on;grid minor;title('AO Parameter Estimation: Current Best','color','w','fontsize',18);
        s(1).YColor = [1 1 1];
        s(1).XColor = [1 1 1];
        s(1).Color  = [.3 .3 .3];
    else
        s(1) = subplot(4,3,1);
        %imagesc(spm_unvec(spm_vec(Y),aopt.yshape));
        if iscell(aopt.yshape)
            surf(spm_unvec(spm_vec(Y),spm_cat(aopt.yshape)),'EdgeColor','none');
        else
            surf(spm_unvec(spm_vec(Y),aopt.yshape),'EdgeColor','none');
        end
        
        title('DATA','color','w','fontsize',18);
        s(1).YColor = [1 1 1];
        s(1).XColor = [1 1 1];
        s(1).ZColor = [1 1 1];
        s(1).Color  = [.3 .3 .3];
        s(6) = subplot(4,3,2);
        %imagesc(spm_unvec(spm_vec(y),aopt.yshape));
        if iscell(aopt.yshape)
            surf(spm_unvec(spm_vec(y),spm_cat(aopt.yshape)),'EdgeColor','none');
        else
            surf(spm_unvec(spm_vec(y),aopt.yshape),'EdgeColor','none');
        end
        
        title('PREDICTION','color','w','fontsize',18);
        s(6).YColor = [1 1 1];
        s(6).XColor = [1 1 1];
        s(6).ZColor = [1 1 1];
        s(6).Color  = [.3 .3 .3];
    end

    %s(2) = subplot(412);
    s(2) = subplot(4,3,[6]);
    %bar([former_error new_error]);
    plot(former_error,'w--','linewidth',3); hold on;
    plot(new_error,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('Error Change','color','w','fontsize',18);
    ylabel('Δ error');
    s(2).YColor = [1 1 1];
    s(2).XColor = [1 1 1];
    s(2).Color  = [.3 .3 .3];
    
    
    %s(3) = subplot(413);
    s(3) = subplot(4,3,7);
    bar(real([ x(:)-ox(:) ]),'FaceColor',[1 .7 .7],'EdgeColor','w');
    title('Parameter Change','color','w','fontsize',18);
    ylabel('Δ prior');
    ylim([-1 1]);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s(3).YColor = [1 1 1];
    s(3).XColor = [1 1 1];
    s(3).Color  = [.3 .3 .3];
    drawnow;
    
    s(4) = subplot(4,3,[4 5]);
    plot(spm_vec(Y),'w:','linewidth',3);
    hold on;
    plot(spm_vec(Y)-aopt.oerror,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;
    title('Last Best','color','w','fontsize',18);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s(4).YColor = [1 1 1];
    s(4).XColor = [1 1 1];
    s(4).Color  = [.3 .3 .3];
    drawnow;    
end

aopt.oerror = new_error;
params.aopt = aopt;

end

% Subfunctions: Fetching states & the objective function
%==========================================================================

function [Y,y] = GetStates(x,params)
% - evaluates the model and returns it along with the stored data Y
%

%global aopt
aopt = params.aopt;

IS = aopt.fun;
P  = x(:)';

try    y  = IS(spm_unvec(P,aopt.x0x0)); 
catch; y  = spm_vec(aopt.y)*0;
end
Y  = aopt.y;

end

function [e,J] = obj_J(x,params)
% wrapped version of objective that returns the full (i.e. MIMO) version of
% the Jacobian
params.aopt.mimo=1;

if nargout==1
    e = obj(x,params);
    J = [];
else
    [e,J,er,mp,Cp,L,params] = obj(x,params);
    J = params.aopt.J;
end
end

function [e,J,er,mp,Cp,L,params] = obj(x0,params)
% Computes the objective function - i.e. the Free Energy or squared error to 
% minimise. Also returns the parameter Jacobian, error (vector), model prediction
% (vector) and covariance
%

aopt = params.aopt;
method = aopt.ObjectiveMethod;

IS = aopt.fun;
P  = x0(:)';

% Evalulate f(X)
%--------------------------------------------------------------------------
warning off
try    y  = IS(spm_unvec(P,aopt.x0x0)); 
catch; y  = spm_vec(aopt.y)*0+inf;
end
warning on;

% Data & precision
%--------------------------------------------------------------------------
Y  = aopt.y;
Q  = aopt.Q;

% Feature selection
%--------------------------------------------------------------------------
if isfield(params,'FS')
    yfs = params.FS(spm_unvec(y,Y));
    Yfs = params.FS(Y);
    
    y = spm_vec(yfs);
    Y = spm_vec(Yfs);
    
    if isnumeric(Q) && ~isempty(Q)
        n1 = length(y) - length(Q);
        Q(end+1:end+n1,end+1:end+n1) = eye(n1); % pad out Q
    end
end

% Manipulations of the objective function for spectral fitting
%--------------------------------------------------------------------------
% includepeaksinerror = aopt.includepeaksinerror;
% if includepeaksinerror
%     % compute euclidean distances between peaks in the output - e.g. when
%     % the user fun returns a power spectrum that we're fitting. Use sum of
%     % these distances as a weighting on the log evidence
%     
%     [~,YI] = findpeaks(real(Y));
%     [~,yi] = findpeaks(real(y));
%     
%     DX = cdist(YI,yi);
%     D  = min(DX,[],2);
%     
%     if any(D)
%         iD     = 1./(D+1);
%         totalD = sum(iD);
%     else
%         totalD = 1;
%     end
%     
% end

% Check / complete the derivative matrix (for the covariance)
%--------------------------------------------------------------------------
if ~isfield(aopt,'J')
    aopt.J = ones(length(x0),length(spm_vec(y)));
end
if isfield(aopt,'J') && isvector(aopt.J) && length(x0) > 1
    %aopt.J = spm_inv(aest_cov(aopt.J,length(aopt.J))) * aopt.J;
    aopt.J = repmat(aopt.J,[1 length(spm_vec(y))]);
end

% Free Energy Objective Function: F(p) = log evidence - divergence
%--------------------------------------------------------------------------
if isnumeric(Q) && ~isempty(Q) 
    % If user supplied a precision matrix, store it so that it can be
    % incorporated into the updating q
    aopt.precisionQ = Q;
end

if ~isfield(aopt,'precisionQ')
    Q  = spm_Ce(1*ones(1,length(spm_vec(y)))); %
    ny  = length(spm_vec(y));
    nq  = ny ./ length(Q);
elseif isfield(aopt,'precisionQ')
    Q   = {aopt.precisionQ};
    %clear Q;
    %lpq = length(aopt.precisionQ);
    %for ijq = 1:length(aopt.precisionQ)
    %    Q{ijq} = sparse(ijq,ijq,aopt.precisionQ(ijq,ijq),lpq,lpq);
    %end
    ny  = length(spm_vec(y));
    nq  = ny ./ length(Q{1});
end

if ~isfield(aopt,'h');
    h  = sparse(length(Q),1) - log(var(spm_vec(Y))) + 4;
else
    h = aopt.h;
end

if isinf(h)
    h = 1/8;
end

iS = sparse(0);

for i  = 1:length(Q)
    iS = iS + Q{i}*(exp(-32) + exp(h(i)));
end

e   = (spm_vec(Y) - spm_vec(y)).^2;
e   = (spm_vec(Y) - spm_vec(y));
ipC = aopt.ipC;

warning off;                                % suppress singularity warnings
Cp  = spm_inv( (aopt.J*iS*aopt.J') + ipC );
warning on

if aopt.rankappropriate
    N = rank((Cp)); % cov rank
    [v,D] = eig(real(mean(e)) + (Cp)); % decompose covariance matrix
    DD  = diag(D); [~,ord]=sort(DD,'descend'); % sort eigenvalues
    Cp = v(:,ord(1:N))*D(ord(1:N),ord(1:N))*v(:,ord(1:N))';
end

p  = ( x0(:) - aopt.pp(:) );

if any(isnan(Cp(:))) 
    Cp = Cp;
end

if aopt.hyperparameters
    if isfield(params.aopt,'h') && ~isempty(params.aopt.h) && params.aopt.computeh
        %fprintf('using precomputed hyperparam while differentiating\n');
        h = params.aopt.h;
        ihC = params.aopt.ihC;
        d = params.aopt.d;
        Ch = params.aopt.Ch;
    else
        %fprintf('\n\n\ncomputing h\n\n\n');
        % pulled directly from SPM's spm_nlsi_GN.m ...
        % ascent on h / precision {M-step}
        %==========================================================================
        clear P;
        nh  = length(Q);
        S   = spm_inv(iS);
        ihC = speye(nh,nh)*exp(4);
        hE  = sparse(nh,1) - log(var(spm_vec(Y))) + 4;
        for i = 1:nh
            P{i}   = Q{i}*exp(h(i));
            PS{i}  = P{i}*S;
            P{i}   = kron(speye(nq),P{i});
            JPJ{i} = real(aopt.J*P{i}*aopt.J');
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

        if aopt.updateh
            aopt.h = h;
            aopt.JPJ = JPJ;
            aopt.Ch  = Ch;
            aopt.d   = d;
            aopt.ihC = ihC;
        end
    end
end % end of if hyperparams (from spm) ... 

% FREE ENERGY TERMS
%==========================================================================

% (1) Complexity minus accuracy of states / observations
%--------------------------------------------------------------------------
L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2;           

% (2) Complexity minus accuracy of parameters
%--------------------------------------------------------------------------
L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;

if aopt.hyperparameters
    % (3) Complexity minus accuracy of precision
    %----------------------------------------------------------------------
    L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2; 
end

if aopt.corrweight
    L(1) = L(1) * corr(spm_vec(Y),spm_vec(y)).^2;
    %sQ   = smooth(diag(aopt.precisionQ));
    %QCx  = atcm.fun.wcor([real(spm_vec(Y)),real(spm_vec(y))],sQ./sum(sQ));
    %L(1) = L(1) * QCx(1,2).^2;
end

try aopt.Cp = Cp;
catch
    aopt.Cp = Cp;
end
    aopt.iS = iS;

params.aopt = aopt;
       
switch lower(method)
    case {'free_energy','fe','freeenergy','logevidence'};
        F    = sum(L);
        e    = (-F);

        if strcmp(lower(method),'logevidence')
            % for log evidence, ignore the parameter term
            % its actually still an SSE measure really
            if ~aopt.hyperparameters
                F = L(1);
                e = -F;
            else
                F = sum( L([1 3]) );
                e = -F;
            end
        end
    
        % Other Objective Functions
        %------------------------------------------------------------------ 
        case 'sse'
            % sse: sum of error squared
            e  = sum( (spm_vec(Y) - spm_vec(y) ).^2 ); e = abs(e);
            
        case 'sse2' % sse robust to complex systems
            e  = sum(sum( ( spm_vec(Y)-spm_vec(y)').^2 ));
            e  = real(e) + imag(e);
    
        case 'mse'
            % mse: mean squared error
            e = (norm(spm_vec(Y)-spm_vec(y),2).^2)/numel(spm_vec(Y));

        case 'rmse'
            % rmse: root mean squaree error
            er = spm_vec(Y)-spm_vec(y);
            %er = er + atcm.fun.adct(er);
            e  = ( (norm(er,2).^2)/numel(spm_vec(Y)) ).^(1/2);
            
            %e = ( (norm(Q{:}*(spm_vec(Y) - spm_vec(y)),2).^2)/numel(spm_vec(Y)) ).^(1/2);
            
        case {'correlation','corr','cor','r2'}
            % 1 - r^2 (bc. minimisation routine == maximisation)
            e = 1 - ( corr( spm_vec(Y), spm_vec(y) ).^2 );

        case 'combination'
            % combination:
            SSE = sum( ((spm_vec(Y) - spm_vec(y)).^2)./sum(spm_vec(Y)) );
            R2  = 1 - abs( corr( spm_vec(Y), spm_vec(y) ).^2 );
            e   = SSE + R2;
            
        case 'euclidean'
        
            ED = cdist(spm_vec(Y),spm_vec(y));
            ED = ED*iS*ED';
            e  = sum(spm_vec(ED)).^2;
        

        case {'logistic' 'lr'}
            % logistic optimisation 
            e = -( spm_vec(Y)'*log(spm_vec(y)) + (1 - spm_vec(Y))'*log(1-spm_vec(y)) );
end

% Error along output vector (when fun is a vector output function)
er = ( spm_vec(y) - spm_vec(Y) ).^2;
mp = spm_vec(y);


% This wraps obj to return only the third output for MIMOs
% when we want the derivatives w.r.t. each output of fun
function er  = inter(x0,params)
    [~,~,~,er] = obj(x0,params);
end

J = [];

% This hands off to jaco, which computes the derivatives
% - note this can be used differently for multi-output systems
if nargout == 2 || nargout == 7
    V    = aopt.pC;
    Ord  = aopt.order; 
    
    % Switch for different steps in numerical differentiation
    if aopt.fixedstepderiv == 1
        V = (~~V)*1e-3;
    elseif aopt.fixedstepderiv == -1
        V = V + ( (~~V) .* abs(randn(size(V))) );
    elseif aopt.fixedstepderiv > 1
        V = (~~V)*aopt.fixedstepderiv;
    end
    
    %aopt.computeiCp = 0; % don't re-invert covariance for each p of dfdp
    params.aopt.updateh = 0;
    params.aopt.computeh = 0;
    params.aopt.h   = h;
    try
        params.aopt.ihC = ihC;
        params.aopt.d   = d;
        params.aopt.Ch  = Ch;
    end
    
    % Switch the derivative function: There are 4: 
    %
    %           mimo | ~mimo
    %           _____|_____
    % parallel |  4     2
    % ~ para   |  3     1
    %
    
    if ~aopt.mimo 
        if ~aopt.parallel
            % option 1: dfdp, not parallel, 1 output
            %----------------------------------------------------------
            [J,ip] = jaco(@obj,x0,V,0,Ord,[],{params});... df[e]   /dx [MISO]
            %objfunn = @(x) obj(x,params);[J,~] = spm_diff(objfunn,x0,1);
        else
            % option 2: dfdp, parallel, 1 output
            %----------------------------------------------------------
            [J,ip] = jacopar(@obj,x0,V,0,Ord,{params});
        end
    elseif aopt.mimo == 1
        nout   = 3;
        if ~aopt.parallel
            % option 3: dfdp, not parallel, ObjF has multiple outputs
            %----------------------------------------------------------
            [J,ip] = jaco_mimo(@obj,x0,V,0,Ord,nout,{params});
        else
            % option 4: dfdp, parallel, ObjF has multiple outputs
            %----------------------------------------------------------
            [J,ip] = jaco_mimo_par(@obj,x0,V,0,Ord,nout,{params});
        end
        J0     = cat(2,J{:,1})';
        J      = cat(2,J{:,nout})';
        J(isnan(J))=0;
        J(isinf(J))=0;
        JI = zeros(length(ip),1);% Put J0 in full space
        JI(find(ip)) = J0;
        J0  = JI;
        
    elseif aopt.mimo == 2
        
        % dpdxdxdx - or approx the curvature of the curvature if order=2
        if ~aopt.parallel
            jfun = @(x0,V) jaco_mimo(@obj,x0-exp(V).*jaco(@obj,x0,V,0,Ord,[],{params}),V,0,Ord,4,{params});
        else
            jfun = @(x0,V) jaco_mimo_par(@obj,x0-exp(V).*jacopar(@obj,x0,V,0,Ord,{params}),V,0,Ord,4,{params});
        end
        [J,ip] = jfun(x0,V);
        J0     = cat(2,J{:,1})';
        J      = cat(2,J{:,4})';
        J(isnan(J))=0;
        J(isinf(J))=0;
        JI = zeros(length(ip),1);
        JI(find(ip)) = J0;
        J0  = JI;        
                
    end
    
    % Embed J in full parameter space
    IJ = zeros(length(ip),size(J,2));
    try    IJ(find(ip),:) = J;
    catch; IJ(find(ip),:) = J(find(ip),:);
    end
    J  = IJ;
    
    % unstable parameters can introduce NaN and inf's into J: set gradient
    % to 0 and it won't get updated...
    %J(isnan(J))=0;
    %J(isinf(J))=0;
    
    aopt.updateh  = 1;
    aopt.computeh = true;
    
    % Store for objective function
    if  aopt.updatej
        aopt.J       = J;
        aopt.updatej = false;     % (when triggered always switch off)
    end
    
    % Accumulate gradients / memory of gradients
    if aopt.memory
        try
            J       = ( J + aopt.pJ ) ./2;
            aopt.pJ = J;
        catch
            aopt.pJ = J;
        end
    end
    
    if aopt.mimo && (aopt.mimo~=3)
        J = spm_vec(J0);
    end
    params.aopt = aopt;
    
end

end

function J = compute_step_J(df0,red,e0,step_method,params,x0,x3,df1)
% Wrapper on the function below to just return the second output!
    [x3,J] = compute_step(df0,red,e0,step_method,params,x0,x3,df1);
end

function [x3,J] = compute_step(df0,red,e0,step_method,params,x0,x3,df1)
% Given the gradients (df0) & parameter variances (red)
% compute the step, 'a' , in:
%
%  dx = x + a*-J
%
% in most cases this is some linear transform of red.
%
aopt = params.aopt;
search_method = step_method;

% J = -gradient
J      = -df0';

switch search_method
    case 1
        
        if aopt.factorise_gradients
            a = ones(size(J,2),1);
            [L,D] = ldl_smola(J',a);
            dFdpp = -(L*(D./sum(diag(D)))*L');
        else
            dFdpp  = -(J'*J);
        end
        
        % Compatibility with older matlabs
        x3  = repmat(red,[1 length(red)])./(1-dFdpp);
        
        [uu,ss,vv] = spm_svd(x3);
        nc = min(find(cumsum(diag(full(ss)))./sum(diag(ss))>=.95));
        x3 = full(uu(:,1:nc)*ss(1:nc,1:nc)*vv(:,1:nc)');
                        
    case 2
        
        J     = -df0';
        dFdp  = -real(aopt.J*aopt.iS*repmat(e0,[ length(aopt.iS),1]) ) - aopt.ipC*aopt.pp;
        dFdpp = -real(aopt.J*aopt.iS*aopt.J')  - aopt.ipC;
        
        % bit hacky but means i don't have to change the function
        x3 = dFdp;
        J  = dFdpp;

    case 3
        
        if aopt.factorise_gradients
            a = ones(size(J,2),1);
            [L,D] = ldl_smola(J',a);
            dFdpp = -(L*(D./sum(diag(D)))*L');
        else
            dFdpp  = -(J'*J);
        end
        
        %Initial step 
        x3  = (4*red)./(1-dFdpp);
        
    case {4 5}
        
        %x3 = (1/64);
        x3 = red(:);
        
     case 6
        % low dimensional hyperparameter tuning based on clustering of the
        % jacobian (works better if derivatives are computed as if the
        % system is not a mimo - i.e.not elementwise on the output function)
        
        
        if aopt.factorise_gradients
            a = ones(size(J,2),1);
            [L,D] = ldl_smola(J',a);
            dFdpp = -(L*(D./sum(diag(D)))*L');
        else
            dFdpp  = -(J'*J);
        end
        
        J = dFdpp;
        
        Cp = params.aopt.Cp;
        Cp(isnan(Cp))=0;
        Cp(isinf(Cp))=0;
        [u,s] = eig(Cp);
        s=diag(s);
        [~,order] = sort(abs(s),'descend');
        r = atcm.fun.findthenearest(cumsum(abs(s(order)))./sum(abs(s(order))),0.99);
        r = round(r);
        
        r = min(r,6);
        r = max(r,2); % make sure its not 0
        
        if isempty(r)
            r = 2;
        end
        
        % restricting r to a very low dimensional (sub)space is conceptually like a
        % variational autoencoder in the assumption that some reduced space
        % can be identified but we can still project back out to full param
        % space
        
        v  = clusterdata(real(J),r);
        try
            V  = sparse(v,1:length(red),1,r,length(red));
            p  = ones(1,r);
            V  = V.*repmat(~~red(:)',[r 1]);

            % options for the objective function
            aopt.updatej = true; aopt.updateh = true; params.aopt  = aopt;

            % hyperparameter tuning to find a in x + a*-J' using parameter
            % components - i.e. x + (p*V)*-J'
            g = @(a) real(obj(x0 + (a*V)'.*J,params));

            if params.aopt.parallel
                options = optimset('Display','off','UseParallel',true);
            else
                options = optimset('Display','off');
            end

            % we're not looking for a global solution so restrict n-its and
            % search space
            options.MaxIter = 50;

            LB = p - 10;
            UB = p + 10;

            p = fmincon(g,p,[],[],[],[],LB,UB,[],options);
            %p = fminsearch(g,p,options);
        
        catch
            V = eye(length(red));
            p = ones(1,length(red));
        end
            
        x3 = (p*V)';
        %J  = -df1;
        
    case 7
        
        if aopt.factorise_gradients
            a = ones(size(J,2),1);
            [L,D] = ldl_smola(J',a);
            dFdpp = -(L*(D./sum(diag(D)))*L');
        else
            dFdpp  = -(J'*J);
        end
        
        % Compatibility with older matlabs
        x3  = repmat(red,[1 length(red)])./(1-dFdpp);
        
        x3(isnan(x3))=0;
        x3(isinf(x3))=0;
        
        %Leading SVD components
        [u,s] = eig(x3);
        [~,order] = sort(abs(diag(s)),'descend');

        s = diag(s);
        s = s(order);
        u = u(:,order);
        
        % Number of components (trajectories) needed
        nc90 = findthenearest(cumsum(abs(s))./sum(abs(s)),0.9);
        xbar = x3*0;
                
        for thisn = 1:nc90
            
            this = full(u(:,thisn));
            
            num_needed = findthenearest( cumsum(sort(abs(this),'descend'))./sum(abs(this)), .99);
    
            [~,I]=maxpoints(abs(this),num_needed);
                    
            % Work backwards to reconstruct x3 from important components
            xbar = xbar + x3(:,I)*diag(this(I))*x3(:,I)';
                        
        end
        
        x3   = xbar;        
            
end

end

function dx = compute_dx(x1,a,J,red,search_method,params)
% Given start point x1, step a and (directional) gradient J, compute dx:
%
% dx = x1 + a*J
%
% (note gradient has already been negative signed)

aopt = params.aopt;

if search_method == 1 || search_method == 7
    %dx    = x1 + (a*J');                 % When a is a matrix
    dx  = x1 + (sum(a)'.*J');
    %dx = x1 + (sum(a).*J')';
elseif search_method == 2
    dFdp  = a;
    dFdpp = J;
    ddx   = spm_dx(dFdpp,dFdp,{red})';    % The SPM way
    ddx   = ddx(:);
    dx    = x1 + ddx;    
elseif search_method == 3                % Rasmussen w/ varying p steps
    dx    = x1 + (a.*J);                 
elseif search_method == 4                % Flat/generic descent
    dx    = x1 + (a.*J);
elseif search_method == 5
        %dx = -spm_pinv(-real(J*aopt.iS*J')-aopt.ipC*x1) * ...
        %            (-real(J*aopt.iS*aopt.er)-aopt.ipC*x1);
        
        dfdx = -spm_pinv(-real(J*aopt.iS*J')-aopt.ipC*x1);
        f    = -real(J*aopt.iS*aopt.er)-aopt.ipC*x1;
        
        [dx] = spm_dx(dfdx,f,{-2});
        dx = x1 + dx;
elseif search_method == 6
    % hyperparameter tuned step size
    dx = x1 + a.*J;
    
end


end

function [Pp,dP] = BayesInf(Ep,pE,Cp)
% Returns the conditional probability of parameters, given 
% posteriors, priors and  covariance
%
% AS

Qp = Ep;
Ep = spm_vec(Ep);
pE = spm_vec(pE);

if isstruct(Cp)
    Cp = diag(spm_vec(Cp));
end

dP = spm_unvec(Ep-pE,Qp);

% Bayesian inference {threshold = prior} 
warning('off','SPM:negativeVariance');
dp  = spm_vec(Ep) - spm_vec(pE);
Pp  = spm_unvec(1 - spm_Ncdf(0,abs(dp),diag(Cp)),Qp);
warning('on', 'SPM:negativeVariance');
 
end

function X = DefOpts()
% Returns an empty options structure with defaults
X.step_method = 3;
X.im          = 1;
X.mleselect   = 0;
X.objective   = 'fe';
X.writelog    = 0;
X.order       = 2;
X.min_df      = 0;
X.criterion   = 1e-3;
X.Q           = [];
X.inner_loop  = 10;
X.maxit       = 4;
X.y           = 0;
X.V           = [];
X.x0          = [];
X.fun         = [];
X.DoMLE       = 0;

X.hyperparams  = 1;
X.force_ls     = 0;
X.doplot       = 1;
X.smoothfun    = 0;
X.ismimo       = 0;
X.gradmemory   = 0;
X.doparallel   = 0;
X.fsd          = 1;
X.allow_worsen = 0;
X.doimagesc    = 0;
X.EnforcePriorProb = 0;
X.FS = [];
X.rankappropriate = 0;
X.userplotfun  = [];
X.ssa          = 0;
X.corrweight   = 0;
X.neuralnet    = 0;
X.WeightByProbability = 0;
X.faster = 0;
X.ext_linesearch = 0;
X.factorise_gradients = 1;
end

function parseinputstruct(opts)
% Gets the user supplied options structure and assigns the options to the
% correct variables in the AO base workspace

fprintf('User supplied options / config structure...\n');

def = DefOpts();
opt = fieldnames(def);

% complete the options 
for i = 1:length(opt)
    if isfield(opts,opt{i})
        def.(opt{i}) = opts.(opt{i});
    end
end

% send it back to the caller (AO.m)
for i = 1:length(opt)
    assignin('caller',opt{i},def.(opt{i}));
end

end

function PrintHelp()

fprintf(['AO implements a gradient descent optimisation that incorporates \n' ...
    'curvature information (like a GaussNewton). Each parameter of f() is \n' ...
    'treated as a Gaussian distribution with variance v. Step sizes are controlled \n' ...
    'by the variance term and calculated using standard method. \n' ...
    'Additional constraint options can be included (e.g. Divergence based). \n'  ...
    'When the full gradient prediction doesnt improve the objective, the routine\n' ...
    'picks a subset of parameters that do. This selection is based on the probability\n' ...
    'of the (GD predicted) new parameter value coming from the prior distribution.\n' ...
    '\nIf using the BayesAdjust option = 1, this selection routine entails MLE\n' ...
    'to find the optimum set of parameter steps that maximise their approx joint probabilities.\n'...
    '\nIn model fitting scenarios, the code is set up so that you pass the model\n'...
    'function (fun), parameters and also the data you want to fit. The advantage of\n'...
    'this is that the algo can compute the objective function. This is necessary\n'...
    'if you want to minimise free energy (but also has SSE, MSE, RMSE etc).\n' ...
    '\nOutputs are the posteriors (means), objective value (F), (co)variance (CP),\n' ...
    'posterior probabilities (Pp) and a History structure (Hist) that contains the\n'...
    'parameters, objective values and gradients from each iteration of the algorithm.\n' ...
    '\nThe code makes use of the fabulous SPM toolbox functions for things like\n' ...
    'vectorising and un-vectorising - so SPM is a dependency. This means that\n' ...
    'the data you''re fitting (y in AO(fun,p,v,y) ) and the output of fun(p)\n'...
    'can be of whatever data type you like (vector, matrix, cell etc).\n' ...
    '\nIf you want to speed up the algorithm, change search_method from 3 to 1.\n'...
    'This method extrapolates further and can fit data quite a bit faster \n'...
    '(and with better fits). However, it is also prone to pushing parameters\n'...
    'to extremes, which is often bad in model fitting when you plan to make\n'...
    'some parameter inference.\n']);



end



% Other experiment stuff
%---------------------------------------------



        % (option) Probability constraint
        %------------------------------------------------------------------
%         if BayesAdjust
%             % This probably isn't sensible, use the JPD method instead.
%             % Probabilities of these (predicted) values actually belonging to
%             % the prior distribution as a bound on parameter step
%             % (with arbitrary threshold)
%             ddx = dx - x1;
%             ppx = spm_Ncdf(abs(x0),abs(dx),sqrt(red)); ppx(ppx<.2) = 0.2;
%                     
%             % Parameter update
%             dx = x1 + (ddx.*ppx);
%             
%             % mock some distributions to visualise changes
%             for i = 1:length(x1)
%                 pd(i)   = makedist('normal','mu',abs(x1(i)),'sigma', ( red(i) ));
%                 pr(i,:) = pdf(pd(i),abs(x1(i))-10:abs(x1(i))+10);
%                 po(i,:) = pdf(pd(i),abs(dx(i))-10:abs(dx(i))+10);
%             end
%             BayesPlot(-10:10,pr,po);
%             
%         end

        % (option) Divergence adjustment
        %------------------------------------------------------------------
%         if DivAdjust
%             % This probably isn't a good idea actually
%             % Divergence of the prob distribution 
%             PQ  = pt(:).*log( pt(:)./pdt(:) );  PQ(isnan(PQ)) = 0;
%             iPQ = 1./(1 - PQ);
% 
%             % Parameter update
%             ddx = dx(:) - x1(:);
%             dx  = x1(:) + ddx.*iPQ(:);
%         end

        % (option) Param selection using maximum joint-probability estimates
        %------------------------------------------------------------------
%         JPD = BayesAdjust;
%         if JPD
%             % Compute JPs from rank sorted p(P)s            
%             [~,o]  = sort(pt(:),'descend');
%             for ns = 1:length(pt)
%                 pjpd(ns) = prod( pt(o(1:ns)) ) * ns;    
%             end
%             
%             % Compute 'growth rate' of (prob ranked) parameter inclusions
%             growth(1) = 1;
%             for i = 2:length(pjpd)
%                 growth(i) = pjpd(i) ./ growth(i-1);
%             end
%             
%             % JPDtol is a hyperparameter             
%             % Starting with the highest probability parameter estimate
%             % (resulting from the GD), it asks, what is the effect of
%             % also updating the parameter with the next highest probability
%             % ... and so on until we hit a small threshold param, JPDtol.
%             selpar       = o( find( growth(:) > JPDtol ));            
%             newp         = x1;
%             newp(selpar) = dx(selpar);
%             dx           = newp(:);
%             
%             probplot(pjpd,JPDtol);
%                                     
%         end


%         % Optionally now compute a gradient descent on the step size, a:
%         % -----------------------------------------------------------------
%         % x[p,t+1] = x[p,t] +         a[p]          *-dfdx[p]
%         % 
%         % x[p,t+1] = x[p,t] + ( a[p] + b*-dfda[p] ) *-dfdx[p]
%         %
%         % where b (the step of the step) is fixed at 1e-4
%         %
%         % Note that optimising 'a' isn't just about speeding up the ascent,
%         % because a is also the variance term on the parameters
%         % distributions, this is also optimising the variance...
%         OptimiseStepSize = BTLineSearch;
%                 
%         if OptimiseStepSize && nfun == 1 && obj(dx,params) < obj(x1,params) && search_method ~= 4
%             
%             pupdate(loc,n,0,e1,e1,'grdstp',toc); 
%             aopt.updateh = false;
%             params.aopt  = aopt;
%             br  = red;
%             
%             % Optimise a!
%             % - Do a gradient descent on the step size!        
%             afun = @(red) obj( compute_dx(x1,compute_step(df0,red,e0,search_method),...
%                                     J,red,search_method),params );
% 
%             % Compute the gradient w.r.t steps sizes
%             %[agrad] = jaco(afun,red,(~~red)*1e-4,0,2);
%             [agrad] = jaco(afun,red,red/8,0,1);
%             
%             % condition it
%             agrad(isinf(agrad))=0;
%             
%             % Backtracking Line Search:
%             %    f(p + g(a)*-y) <= f(p + a*-y)
%             
%             chn = 0; loop = true;
%             while loop
%             
%                 chn   = chn + 1;
%                 d_red = red + ( (1e-3)*-agrad );
% 
%                 % Update only stable new step sizes
%                 bad = unique([ find(isnan(d_red)); ...
%                                find(isinf(d_red)); ...
%                                find(d_red < 0) ]);
%                 d_red(bad) = red(bad);
%                 red        = d_red;
% 
%                 % Final ddx
%                 [da,~]  = compute_step(df0,red,e0,search_method);
%                 ddx     = compute_dx(x1,da,J,red,search_method);
%                 
%                 if obj(ddx,params) < obj(dx,params)
%                     if chn == 1
%                         pupdate(loc,n,0,obj(ddx,params),e1,'BkLnSr',toc);
%                     end
%                     dx = ddx;
%                     a  = da;
%                     br = red;   
%                     
%                     % Update the (inverse) variance used in computing F
%                     aopt.ipC   = spm_inv(spm_cat(spm_diag({diag(red)})));
%                 else
%                     red  = br;
%                     loop = false;
%                 end
%                 if chn == 20
%                     loop = 0;
%                 end
%             end
%             
%             aopt.updateh = true;
%             
%         end