function [X,F,Cp,PP,Hist,params] = olderAO(funopts)
% A (Bayesian) gradient (curvature) descent optimisation routine, designed primarily 
% for parameter estimation in nonlinear models.
%
% Getting started with default options:
% 
% op = AO('options')
% 
% op.fun = func;       % function/model f(x0)
% op.x0  = x0(:);      % start values: x0
% op.y   = Y(:);       % data we're fitting (for computation of objective fun, e.g. e = Y - f(x)
% op.V   = V(:);       % variance / step for each parameter, e.g. ones(length(x0),1)/8
% 
% op.objective='gauss'; % select smooth Gaussian error function
% 
% Run the routine:
% [X,F,CV,~,Hi] = AO(op); 
% 
%
% change objective to 'gaussmap' for MAP estimation%
%
% The algorithm combines a Newton-like gradient routine (& optionally
% MAP and ML) with line search and an optimisation on the composition of update 
% parameters from a combination of the gradient flow and memories of previous updates.
% It further implements an exponential cost function hyperparameter which
% is tuned independently, and an iteratively updating precision operator.
% 
% General idea: I have some data Y0 and a model (f) with some parameters x.
% Y0 = f(x) + e  
%
% For a multivariate function f(x) where x = [p1 .. pn]' a gradient descent scheme
% is:
%
%   x[p,t+1] = x[p,t] + a[p] *-dFdx[p]
%
% Note the use of different step sizes, a[p], for each parameter.
% Optionally, low dimensional hyperparameter tuning can be used to find 
% the best step size a[p], by setting step_method = 6;
%
%   x[p,t+1] = x[p,t] + (b*V) *-dFdx[p]  ... where b is optimised independently
%
% See jaco.m for options, although by default these are computed using a 
% finite difference approximation of the curvature, which retains the sign of the gradient:
%
% f0 = F(x[p]+h) 
% fx = F(x[p]  )
% f1 = F(x[p]-h) 
%
% j(p,:) =        (f0 - f1) / 2h  
%          ----------------------------
%            (f0 - 2 * fx + f1) / h^2  
%
% The algorithm computes the objective function itself based on user
% option; to retreive an empty options structure, do:
%
% opts = AO('options')
%
% Compulsory arguments are: (full list of optionals below)
%
% opts.y     = data to fit
% opts.fun   = model or function f(x)  
% opts.x0    = parameter vector (initial guesses) for x in f(x)
% opts.V     = Var vector, with initial variance for each elemtn of x
% opts.objective = the objective function selected from:
% {'sse' 'mse' 'rmse' 'mvgkl' 'gauss' 'gaussmap' 'gaussq' 'jsd' 'euclidean' 'gkld'}
%
% then to run the optmisation, pass the opts structure back into AO with
% these outputs:
%
%  [X,F,Cp,PP,Hist,params] = AO(opts)
%
% Optional "step" methods (def 9: normal fixed step GD):
% -- step_method = 1 invokes steepest descent
% -- step_method = 3 or 4 invokes a vanilla dx = x + a*-J descent
% -- step_method = 6 invokes hyperparameter tuning of the step size.
% -- step_method = 7 invokes an eigen decomp of the Jacobian matrix
% -- step_method = 8 converts the routine to a mirror descent with
%    Bregman proximity term.
%
% By default momentum is included (opts.im=1). The idea is that we can 
% have more confidence in parameters that are repeatedly updated in the 
% same direction, so we can take bigger steps for those parameters as the
% optimisation progresses.
%
% NOTE - the default option [2022] is now to use a regularised Newton routine:
%---------------------------------------------------------------------------
% 
%  dx = x + inv(H*L*H)*-J
%
% whewre H is the Hessian and J the jacobian (dFdx). L is a regularisation
% term that is optimised independently. 
%
% The (best and default) objective function is you are unsure, is 'gauss'
% which is simply a smooth (approx Gaussian) error function, or 'gaussq'
% which is similar to gauss but implements a sort of pca. 
%
% If you want true MAP estimates (or just to be Bayesian), use 'gaussmap'
% which implements a MAP routine: 
%
%  log(f(X|p)) + log(g(p))
%
%
% Other important stuff to know:
% -------------------------------
% if your function f(x) generates a vector output (not a single value),
% then you can compute the partial gradients along each oputput, which is
% necessary for proper implementation of some functions e.g. GaussNewton;
% flag:
%
% opts.ismimo = 1;
%
% The gradient computation can be done in parallel if you have a cluster or
% multicore computer, set:
%
% opts.doparallel = 1;
%
% Set opts.hypertune = 1 to append an exponential cost function to the chosen 
% objective function. This is defined as:
% 
% c = t * exp(1/t * data - pred)
% 
% where t is a (temperature) hyperparameter controlled through a separate gradient
% descent routine.
%
% ALSO SET:
%
% opts.memory_optimise = 1; to optimise the weighting of dx on the gradient flow and recent memories 
% opts.opts.rungekutta = 1; to invoke a runge-kutta optimisation locally around the gradient predicted dx
% opts.updateQ = 1; to update the error weighting on the precision matrix 
%  
% The {fe / free energy} objective function minimised is
%==========================================================================
%  L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2; % complexity minus accuracy of states
%  L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;                            % complexity minus accuracy of parameters
%  L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;                            % complexity minus accuracy of precision (hyperparameter)
%  F    = -sum(L);
%
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
% opts.WeightByProbability = 0; % weight parameter step update by prob
% opts.faster = 0;          % make faster by limiting loops
% opts.factorise_gradients = 1; % factorise gradients
% opts.sample_mvn = 0;      % when  stuck sample from a multivariate gauss
% opts.do_gpr = 0;          % try to refine estimates using a Gauss process
% opts.normalise_gradients=0;% normalise gradients to a pdf
% opts.isGaussNewton = 0;   % explicitly make the update a Gauss-Newton
% opts.do_poly=0;           % same as do_gpr but using polynomials
% opts.steps_choice = [];   % use multiple step methods and pick best on-line
% opts.hypertune  = 1;      % tune a hyperparameter using exponential cost
% opts.memory_optimise = 1; % switch on memory (recommend)
% opts.rungekutta = 1;      % RK line search (recommend)
% opts.updateQ = 1;         % update precision on each iteration(recommend)
% opts.DoMAP = 0;           % maximum a posteriori
% opts.DoMAP_Bayes = 0;     % Bayesian MAP (recommend, make sure mimo=1)
% opts.crit = [0 0 0 0 0 0 0 0];
% opts.save_constant = 0;   % save .mat on each iteration
% opts.nocheck = 0;         % skip some checks to make faster
% opts.isQR = 0;            % use QR decomp to solve for dx 
% opts.NatGrad = 0;         % work in natural gradients
% opts.variance_estimation = 0; % try to estimate variances
% opts.gradtol = 1e-4;      % minimum tol on dFdx
% opts.isGaussNewtonReg=1;  % try to use DampedGaussNewton steps
% opts.orthogradient=1;     % orthogonalise jacobian/partial gradients
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
% Dependencies
%-------------------------------------------------------------------------
% atcm -> https://github.com/alexandershaw4/atcm
% spm  -> https://github.com/spm/
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
aopt.hypertune       = hypertune;
aopt.verbose = verbose;

BayesAdjust = mleselect; % Select params to update based in probability
IncMomentum = im;        % Observe and use momentum data            
givetol     = allow_worsen; % Allow bad updates within a tolerance
EnforcePriorProb = EnforcePriorProb; % Force updates to comply with prior distribution
WeightByProbability = WeightByProbability; % weight parameter updates by probability
ExLineSearch = ext_linesearch;

params.aopt = aopt;      % Need to move form global aopt to a structure
params.userplotfun = userplotfun;

if save_constant
    name = ['optim_' date];
end

% parameter and step / distribution vectors
x0  = full(x0(:));
XX0 = x0;
V   = full(V(:));
v   = V;
pC  = diag(V);

%crit = [0 0 0 0 0 0 0 0];

% variance (in reduced space)
%--------------------------------------------------------------------------
V     = eye(length(x0));    %turn off svd 
pC    = V'*(pC)*V;
ipC   = spm_inv(spm_cat(spm_diag({pC})));
red   = (diag(pC));

aopt.updateh = true; % update hyperpriors
aopt.pC  = red;      % store for derivative & objective function access
aopt.ipC = ipC;      % store ^
red_x0 = red;

% initial probs
aopt.pt = zeros(length(x0),1) + (1/length(x0));

params.aopt = aopt;

% initial objective value (approx at this point as missing covariance data)
[e0]       = obj(x0,params);
n          = 0;
iterate    = true;
Vb         = V;
    
% initial error plot(s)
%--------------------------------------------------------------------------
if doplot
    f = setfig(); params = makeplot(x0,x0,params); aopt.oerror = params.aopt.oerror;
    pl_init(x0,params)
end

% initialise counters
%--------------------------------------------------------------------------
n_reject_consec = 0;
search          = 0;

% Initial probability threshold for inclusion (i.e. update) of a parameter
Initial_JPDtol = 1e-10;
JPDtol = Initial_JPDtol;
etol = 0;

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

all_dx = [];
all_ex = [];
Hist.e = [];
%etol   = 0;

% start optimisation loop
%==========================================================================
while iterate
        
    % counter
    %----------------------------------------------------------------------
    n = n + 1;    tic;
   
    % Save each step if requested
    if save_constant
        save(name,'x0','Hist');
    end
    
    if WeightByProbability
        aopt.pp = x0(:);
    end
    
    % update Gaussian precision hyperparameters matrix
    % integrate error-weighted Q over iterations: Q(t+1) = dt * Q(t) + dQ/de
    %----------------------------------------------------------------------
    if ~isempty(Q) && updateQ
        if verbose; fprintf('| Updating Q...\n'); end
        [~,~,erx,g0]  = obj( V*x0(ip),params );
        
        % extract residuals & convert to Gaussian mixture [gp]
        RSE = erx(1:length(Q));%rescale(real(erx(1:length(Q))));
                
        gerr = AGenQ(RSE);
        % fit gauss residuals to data using lsq
        b=AGenQ(RSE)\RSE-g0(1:length(Q));
        aopt.Q = Q + diag(b'*gerr);
        %aopt.Q = (Q + AGenQ(RSE) );
        Hist.QQ(n,:) = rescale(real(erx(1:length(Q))));
        %s=subplot(5,3,13);plot(real(diag(aopt.Q)),'color',[1 .7 .7],'linewidth',3);
        s=subplot(5,3,12);imagesc(real(Q));
        title('Gaussian Hyperparameter','color','w','fontsize',18);
        ax       = gca;
        ax.XGrid = 'off';ax.YGrid = 'on';
        s.YColor = [1 1 1];s.XColor = [1 1 1];s.Color  = [.3 .3 .3];
        drawnow;
    end
    
    % compute gradients & search directions
    %----------------------------------------------------------------------
    aopt.updatej = true; aopt.updateh = true; params.aopt  = aopt;
    
    if verbose; pupdate(loc,n,0,e0,e0,'gradnts',toc); end
    
    % Second order partial derivates of F w.r.t x0 using Jaco.m
    [e0,df0,~,~,~,~,params]  = obj( V*x0(ip),params );
    [e0,~,er] = obj( V*x0(ip),params );
    df0 = real(df0);
       
    if normalise_gradients
        df0 = df0./sum(df0);
    end
    
    % catch instabilities in the gradient - ie explosive parameters
    df0(isinf(df0)) = 0;
    
    % Update aopt structure and place in params
    aopt         = params.aopt;
    aopt.er      = er;
    aopt.updateh = false;
    params.aopt  = aopt;
                     
    % print end of gradient computation (just so we know it's finished)
    if verbose; pupdate(loc,n,0,e0,e0,'grd-fin',toc); end
    
    % update hyperparameter tuning plot
    if hypertune; plot_hyper(params.hyper_tau,[Hist.e  e0]); end
    
    % update h_opt plot
    if hyperparams; plot_h_opt(params.h_opt); drawnow; end
    
    % Switching for different methods for calculating 'a' / step size
    if autostep; search_method = autostepswitch(n,e0,Hist);
    else;        search_method = step_method;
    end
    
    % initial search direction (steepest) and slope
    %----------------------------------------------------------------------    
    % Compute step, a, in scheme: dx = x0 + a*-J
    if  n == 1
        a = red*0;
    end
    
    % Setting step size to 6 invokes low-dimensional hyperparameter tuning
    if verbose
        if search_method ~= 6
            pupdate(loc,n,0,e0,e0,'stepsiz',toc);
        else
            pupdate(loc,n,0,e0,e0,'hyprprm',toc);
        end
    end
    
    % Feature Scoring for MIMOs    
    %----------------------------------------------------------------------    
    if orthogradient && ismimo
        if verbose; fprintf('Orthogonalising Jacobian\n'); end
        params.aopt.J = symmetric_orthogonalise(params.aopt.J);
    end

    % i.e. where J(np,nf) & nf > 1
    if ismimo
        if verbose; pupdate(loc,n,0,e0,e0,'scoring',toc); end

        JJ = params.aopt.J;
        Q0 = aopt.Q;
        
        if isempty(Q0)
            Q0 = eye(length(y(:)));
        end
        
        padQ = size(JJ,2) - length(Q0);
        Q0(end+1:end+padQ,end+1:end+padQ)=mean(Q0(:))/10;
        Q0 = atcm.fun.gausvdpca(Q0,np);

        for i = 1:np
            for j = 1:np
                % information score / approximate (precision weighted) Hessian
                score(i,j) = trace(JJ(i,:).*Q0.*JJ(j,:)');
            end
        end
        
        % integrate: da(t) = (expm(da/dx*t) - I)*inv(da/dx)*f
        red = spm_dx(score,diag(pC),1);

        % optimise by re-regularising step size if needed
        if any(isnan(red)) || any(isinf(red)) || norm(red) > 1e2
            Reg = 1/8;
            while true
                red = spm_dx(score,diag(pC),Reg);
                if any(isnan(red)) || any(isinf(red)) || norm(red) > 1e2
                    Reg = Reg*Reg;
                else
                    break;
                end
            end
        end

    end

    % Select step size method
    %----------------------------------------------------------------------    
    if ~ismimo
        [a,J,nJ,L,D] = compute_step(df0,red,e0,search_method,params,x0,a,df0);
    else
        [a,J,nJ,L,D] = compute_step(params.aopt.J,red,e0,search_method,params,x0,a,df0);
        J = -df0(:);
    end

    % if simplelinesearch
    %     if verbose; pupdate(loc,n,0,e0,e0,'linesrc',toc); end
    %     fls = @(r) obj(x0 - (red.*r).*df0,params);
    %     a = hypertuner(fls,ones(size(red))/2);
    % end

    if verbose;
        if search_method ~= 6
            pupdate(loc,n,0,e0,e0,'stp-fin',toc);
        else
            pupdate(loc,n,0,e0,e0,'hyp-fin',toc);
        end    
    end
    
    % Log start of iteration (these are returned)
    Hist.e(n) = e0;
    Hist.p{n} = x0;
    Hist.J{n} = df0;
    Hist.a{n} = a;
    
    if ismimo
        Hist.Jfull{n} = aopt.J;
    end
    
    % Make copies of error and param set for inner while loops
    x1  = x0;
    e1  = e0;
        
    % Start counters
    improve = true;
    nfun    = 0;
    
    % check norm of gradients (def gradtol = 1e-4)
    if norm(J) < gradtol
        fprintf('Gradient step below tolerance (%d)\n',norm(J));
        [X,F,Cp,PP] = finishup(V,x0,ip,e0,doparallel,params,J,Ep,red,writelog,loc,aopt);
        return;
    end
    
    % Update expectation on variance (2nd moments) using Jacobian
    %----------------------------------------------------------------------    
    Hist.red(n,:) = red;
    % if variance_estimation
    %     if n > 1
    %         % this is also a prediction of the step size for the next iter
    %         ored = red;            
    % 
    %         for l = 1:np
    %             PR(l) = normcdf(mean(Hist.Jfull{end}(l,:)),mean(Hist.Jfull{end-1}(l,:)),std(Hist.Jfull{end-1}(l,:)));
    %         end
    % 
    %         red = red_x0 + (red_x0(:) .* PR(:));
    % 
    %         % update saved inverse
    %         %params.aopt.ipC   = spm_inv(diag(red));
    %        % aopt.red = red;
    % 
    %         % update plot
    %         s = subplot(5,3,14);
    %         b = bar([red-ored]);
    % 
    %         %b(1).FaceColor = [1 1 1];
    %         b(1).FaceColor = [1 .7 .7];
    % 
    %         title('Step Change','color','w','fontsize',18);
    %         ax       = gca;
    %         ax.XGrid = 'off';
    %         ax.YGrid = 'on';
    %         s.YColor = [1 1 1];
    %         s.XColor = [1 1 1];
    %         s.Color  = [.3 .3 .3];
    %         drawnow;
    % 
    %     end
    % end
        
    


    % iterative descent on this (-gradient) trajectory
    %======================================================================
    while improve
                
        % Log number of function calls on this iteration
        nfun = nfun + 1;
        
        % Compute The Parameter Step (from gradients and step sizes):
        % % x[p,t+1] = x[p,t] + a[p]*-dfdx[p] 
        %------------------------------------------------------------------
        % dx ~ x1 + ( a * J );
        dx   = compute_dx(x1,a,J,red,search_method,params);  

                    
        if (isGaussNewton && ismimo) || (isQuasiNewton && ismimo)
            % [note this is now just a newton method, not gn!]
            %
            % note for this full Newton method you need both Hess and Grad
            % from the jaco_par.m function because the step size is
            % determined by the inverse of the inner prouect of the hessian
            if verbose; pupdate(loc,n,nfun,e1,e1,'Newton ',toc); end
            
            % by using the variance (red) as a lambda on the inverse
            % Hessian, this becomes a relaxed or 'damped' Newton scheme
            for i = 1:size(J,1);
                for j = 1:size(J,1); 
                    H(i,j) = spm_trace(aopt.J(i,:),aopt.J(j,:));
                end
            end

            %H = [aopt.J];
            H = (red.*H./norm(H));%./norm(H);

            if isQuasiNewton
                [u,s0,v0] = svd(H);
                H = pinv(u);
            end
            
            % the non-parallel finite different functions return gradients
            % in reduced space - embed in full vector space
            Jo = cat(1,aopt.Jo{:,1});
            JJ = x0*0;
            JJ(find(diag(pC))) = Jo;
             
            Hstep = spm_dx(H,JJ,{-4});
            
            Gdx = x1 - Hstep;
            
            dx = Gdx;
            if verbose; fprintf('Selected Newton Step\n'); end
            %if obj(Gdx,params) < obj(dx,params) && ~docompare
            %    dx = Gdx;
            %    fprintf('Selected Newton Step\n');
            %end
        end
        
        if isGaussNewtonReg && ismimo
            % [note this is now just a newton method, not gn!]
            %
            % note for this full Newton method you need both Hess and Grad
            % from the jaco_par.m function because the step size is
            % determined by the inverse of the inner prouect of the hessian
            if verbose; pupdate(loc,n,nfun,e1,e1,'Newton ',toc);end
            
            % by using the variance (red) as a lambda on the inverse
            % Hessian, this becomes a relaxed or 'damped' Newton scheme
            for i = 1:size(J,1);
                for j = 1:size(J,1); 
                    H(i,j) = spm_trace(aopt.J(i,:),aopt.J(j,:));
                end
            end

            %H = [aopt.J];
            H = (red.*H./norm(H));%./norm(H);
            
            % the non-parallel finite different functions return gradients
            % in reduced space - embed in full vector space
            Jo = cat(1,aopt.Jo{:,1});
            if length(Jo) ~= length(x1)
                JJ = x0*0;
                JJ(find(diag(pC))) = Jo;
            else
                JJ = Jo;
            end
            
            % essentially here we are tiuning this part of the Newton
            % scheme:
            %                ______
            % xhat = x - inv(H*L*H')*J

            % tunable regularisation function
            Gf  = @(L) pinv(H*(L*eye(length(H)))*H');
            Gff = @(x) obj(x1 - Gf(x)*JJ,params);
            
            [XX] = fminsearch(Gff,1);
            
            Hstep = pinv(H*(XX*eye(length(H)))*H')*JJ;
                        
            GRdx = x1 - Hstep;
            %dx = GRdx;
            % only take a Newton step if its better than the vanilla
            % gradient step
            
            if obj(GRdx,params) < obj(dx,params) && ~docompare
                dx = GRdx;
                if verbose; fprintf('Selected Regularised Newton Step\n'); end
            end
            
            if forcenewton
                dx = GRdx;
                if verbose; fprintf('Forced Newton Step\n');end
            end
            
        end
            
        % if NatGrad
        %     % retrieve derivatives
        %     if ~ismimo;  j  = J(:)*er';
        %     else;        j = aopt.J;
        %     end
        % 
        %     lambda = 1/8;
        %     D = size(j,1);
        %     % compute natural gradient
        %     fim = inv(lambda*eye(D)+j*j');
        %     dx = x1 - sum(fim*j,2);
        % end
        % 
        % if isQR && ismimo
        %     if verbose; pupdate(loc,n,nfun,e1,e1,'QRmodel',toc); end
        %     % Use QR factorisation to estimate dx from jacobian and
        %     % residual - i.e. under assumption local-linearity 
        %     % (ensure ismimo = 1)
        % 
        %     % note this is the QR equivalent of the normal equations for a
        %     % linear least squares problem
        %     [q,r] = qr(params.aopt.J);
        %     dx = q\((r./norm(r))*er);
        % end
        
        if lsqjacobian
            jx = aopt.J'\y;
            dx = x1 - jx;
        end

        
        % The following options are like 'E-steps' or line search options 
        % - i.e. they estimate the missing variables (parameter indices & 
        % values) that should be optimised in this iteration
        %==================================================================
        
        % Compute the probabilities of each (predicted) new parameter
        % coming from the same distribution defined by the prior (last best)        
        dx  = real(dx);
        x1  = real(x1);
        red = real(red);
        
        % [Prior] distributions
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
        
        % Compute relative change
        pdx = pt*0;
        for i = 1:length(x1)
            if red(i)
                %vv     = real(sqrt( red(i) ));
                vv     = real(sqrt( red(i) ))*2;
                if vv <= 0 || isnan(vv) || isinf(vv); vv = 1/64; end
                pd(i)  = makedist('normal','mu', real(aopt.pp(i)),'sigma', vv);
                pdx(i) = normcdf(dx(i),pd(i).mu,pd(i).sigma);
                %pdx(i) = (1./(1+exp(-pdf(pd(i),dx(i))))) ./ (1./(1+exp(-pdf(pd(i),aopt.pp(i)))));
            else
            end
        end
        pt = pdx;
        prplot(pt);
        aopt.pt = [aopt.pt pt(:)];
        
        % If WeightByProbability is set, use p(dx) as a weight on dx
        % iteratively until n% of p(dx[i]) are > threshold
        % -------------------------------------------------------------
        if WeightByProbability
            dx = x1 + ( pt(:).*(dx-x1) );
            
            if verbose; pupdate(loc,n,1,e1,e1,'OptP(p)',toc); end
            
            optimise = true;
            num_optloop = 0;
            while optimise
                pdx = pt*0;
                num_optloop = num_optloop + 1;
                
                for i = 1:length(x1)
                    if red(i)
                        vv     = real(sqrt( red(i) ))*2;
                        if vv <= 0 || isnan(vv) || isinf(vv); vv = 1/64; end
                        pd(i)  = makedist('normal','mu', real(aopt.pp(i)),'sigma', vv);
                        
                        pdx(i) = normcdf(dx(i),pd(i).mu,pd(i).sigma);
                        %pdx(i) = (1./(1+exp(-pdf(pd(i),dx(i))))) ./ (1./(1+exp(-pdf(pd(i),aopt.pp(i)))));
                    else
                    end
                end
                
                % integrate (update) dx
                dx = x1 + ( pt(:).*(dx-x1) );
                
                % convergence
                if length(find(pdx(~~red) > 0.8))./length(pdx(~~red)) > 0.7 || num_optloop > 2000
                    optimise = false;
                end
                
            end
            
        end
        
        % Save for computing gradient ascent on probabilities
        p_hist(n,:) = pt;
        Hist.pt(:,n)  = pt;
        
        % Update plot: probabilities
        [~,oo]  = sort(pt(:),'descend');
        %probplot(cumprod(pt(oo)),0.95,oo);
        
        % This is a variation on the Gauss-Newton algorithm - compute
        % MLE via WLS - where the weights are the probabilities
        %------------------------------------------------------------------
        % if DoMLE
        %     iter_mle = true;
        %     nmfun    = 0;
        %     clear b
        % 
        %     while iter_mle
        %         if verbose; pupdate(loc,n,nmfun,e1,e1,'MLE/WLS',toc);end
        %         nmfun = nmfun + 1;
        %         nfun  = nfun  + 1;
        % 
        %         % parameter derivatives
        %         if ~ismimo;  j  = J(:)*er';
        %         else;        j = aopt.J;
        %         end
        % 
        %         % weights and residuals
        %         w  = pt;
        %         r0 = spm_vec(y) - spm_vec(params.aopt.fun(spm_unvec(x1,aopt.x0x0)));
        % 
        %         if ~isempty(FS)
        %             r0 = FS(r0);
        %         end
        % 
        %         db  = ( pinv(j'*diag(w)*j)*j'*diag(w) )'*r0;
        % 
        %         % b(s+1) = b(s) - (J'*J)^-1*J'*r(s)
        %         try    b = b - db;
        %         catch; b = db;
        %         end
        % 
        %         % inclusion of a weight essentially makes this a Marquardt regularisation parameter
        %         if isvector(a)
        %             dxd = x1 - a.*b;
        %         else
        %             dxd = x1 - a*b; % recompose dx including step matrix (a)
        %         end
        % 
        %         % update x
        %         if obj(dxd,params) < obj(x1,params) && (nmfun < 3)
        %             x1 = dxd;
        %             x0 = x1;
        %             e1 = obj(x1,params);
        %             e0 = e1;
        %         else
        %             iter_mle = false;
        %             dx = dxd;
        %         end
        %     end
        % end
        % 
        % if DoMAP
        %     % Do MAP estimation of parameter values with lambda as a
        %     % hyperparameter
        % 
        %     if verbose; pupdate(loc,n,n,e1,e1,'MAP est',toc); end
        % 
        %     % retrieve derivatives
        %     if ~ismimo;  j  = J(:)*er';
        %     else;        j = aopt.J;
        %     end
        % 
        %     % remove extra stuff in Jacobian added by feature selection
        %     j = j(:,1:size(y,1));
        % 
        %     lambda = 1/8;
        %     D = size(j,1);
        % 
        %     % try to force a wanring suppression within anonym func call
        %     warning('off','MATLAB:singularMatrix')
        %     wn = @() warning('off','MATLAB:nearlysingularMatrix');
        %     wwn = @() suppress(wn);
        %     warning('off','MATLAB:curvefit:fit:noStartPoint');
        % 
        %     map  = @(lambda) [feval(wwn) (lambda*eye(D)+j*j')\(j*y)];
        %     gmap = @(lambda) obj(x1 - red.*map(lambda),params);
        % 
        %     % optimise - find lambda
        %     ddx = fminsearch(gmap,1/8);
        % 
        %     % recover dx
        %     dx = x1 - red.*map(ddx);
        % 
        % end
        % 
        % if DoMAP_Bayes
        %     % Fully Bayesian MAP estimation of parameter values with
        %     % lambda as an optimisable hyperparameter
        %     % -- also provides an updated expectation for the variance
        % 
        %    if verbose; pupdate(loc,n,nfun,e1,e1,'bMAPest',toc); end
        % 
        %     % retrieve derivatives
        %     if ~ismimo;  j  = J(:)*er';
        %     else;        j = aopt.J;
        %     end
        % 
        %     % remove extra stuff in Jacobian added by feature selection
        %     j = j(:,1:size(y,1));
        % 
        %     % largest eigenvalue normalisation
        %     [ev,ei] = eig(j*j');
        %     j = j ./ max(diag(ei));
        % 
        %     lambda = 1/8;
        % 
        %     D = size(j,1);
        % 
        %     % inject warning silence to anon function 'map'
        %     warning('off','MATLAB:singularMatrix')
        %     wn = @() warning('off','MATLAB:nearlysingularMatrix');
        %     wwn = @() suppress(wn);
        %     warning('off','MATLAB:curvefit:fit:noStartPoint');
        % 
        %     % compute residual and weight
        %     r0 = spm_vec(y) - spm_vec(params.aopt.fun(spm_unvec(x1,aopt.x0x0)));
        %     if length(aopt.iS) == length(r0)
        %         r0 = r0.*aopt.iS.*r0';
        %         r0 = r0(1:size(j,2),1:size(j,2));
        %     else
        %         r0 = r0.*aopt.iS(1:length(r0),1:length(r0)).*r0';
        %     end
        %     r0 = diag(r0);
        % 
        %     % MAP projection using Gauss-Newton / Regularised Levenberg-marquardt
        %     map  = @(lambda) [feval(wwn) (lambda*eye(D)+j*j')\(j*r0)];
        %     gmap = @(lambda) obj(x1 - red.*map(lambda),params);
        % 
        %     % optimise - find lambda
        %     ddx = fminsearch(gmap,1/8);
        % 
        %     % recover dx
        %     Bdx = x1 - red.*map(ddx);
        % 
        %     if obj(Bdx,params) < obj(dx,params) && ~docompare
        %         dx = Bdx;
        %     end
        % 
        %     % recover (co)variance
        %     %if variance_estimation
        %     %    cv = inv((ddx/2)^(-2)*(j*j')+(ddx/2)^(-2)*eye(D));
        %     %    %cv = cv ./ norm(cv);
        %     %    red = 2*sqrt(diag(cv));
        %     %    aopt.pC     = red;
        %     %    params.aopt = aopt;
        %     %end
        % 
        %     %dx = x1 - cv*map(ddx);
        %     %dx = x1 - ((lambda*eye(D)+pinv(cv))+j*j')\(j*r0);
        % 
        %     % note.
        %     % inv(lambda*eye(D)+j*j') is the Fisher-information matrix
        % end
        % 
        % if docompare
        % 
        %     if verbose; pupdate(loc,n,nfun,e1,e1,'compare',toc);end
        % 
        %     OP = [dx(:) Gdx(:) Bdx(:) GRdx(:)];
        %     K  = [1 1 1 1]/4;
        % 
        %     if params.aopt.parallel
        %         options = optimset('Display','off','UseParallel',true);
        %     else
        %         options = optimset('Display','off');
        %     end
        % 
        %     gx = @(K) obj(OP*K',params);
        %     X  = fminsearch(gx,K,options);
        %     dx = OP*X';
        % 
        % end
                
                            
        % (option) Momentum inclusion
        %------------------------------------------------------------------
        if n > 2 && IncMomentum
            if verbose; pupdate(loc,n,nfun,e1,e1,'momentm',toc); end
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
        
        
        if (obj(dx,params) < obj(x1,params) && ~BayesAdjust && ~aopt.forcels) || nocheck
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

                % Could do some scoring here
                %pupdate(loc,n,nfun,e1,e1,'scoring',toc);
                %REL = real(e1 - DFE);
                %red = rescale(REL./norm(REL),-1,1)/8;
                %dx = x1 + (red.*dx);
                
                if givetol;
                            etol = 1./1+exp(1./(n))/(maxit*2);
                else;       etol = 0;
                end
                
                % Identify improver-parameters
                if nfun == 1
                    gp  = double(DFE <= (e0+abs(etol))); % e0
                else
                    gp  = double(DFE <= (e1+abs(etol))); % e0
                end
                
                gpi = find(gp);
                
                if isempty(gp)
                    gp  = ones(1,length(x0));
                    gpi = find(gp);
                    DFE = ones(1,length(x0))*de;
                end
                
            end
            
            %if ~BayesAdjust
                % If the full gradient prediction over parameters did not
                % improve, but the BayesAdjust option is not selected, then
                % only update parameters showing improvements in objective func
                ddx        = V*x0;
                ddx(gpi)   = dx(gpi);
                dx         = ddx;
                de         = obj(dx,params);
                
            % else
            %     % MSort of maximum likelihood - opimise p(dx) according to
            %     % initial conditions (priors; a,b) and error
            %     % arg max: p(dx | a,b & e)
            % 
            %     alpha = 0.95;
            %     thresh = 1 - alpha ;
            % 
            %     if n>1
            %         thresh = 1 - (alpha - (1-(mean(1 - (p_hist(end,:)-p_hist(end-1,:))))) );
            %     end
            % 
            %     if verbose; pupdate(loc,n,nfun,e1,e1,'mleslct',toc);end
            % 
            %     PP = pt(:) ;
            % 
            %     % parameter selection based on probability:
            %     % sort h2l th probabilities and find jpdtol intersect
            %     [~,o]  = sort(PP(:),'descend');
            % 
            %     epar = e0 - DFE(o);
            %     px   = cumprod(PP(o));
            % 
            %     pI   = find( px > thresh );
            %     [~,pnt] = min( epar(pI) );
            % 
            %     selpar = o(1:pnt); % activate parameters
            % 
            %     newp         = x1;
            %     newp(selpar) = dx(selpar);
            %     %if nfun>1 && all(newp(:)~=dx(:))
            %     dx           = newp(:);
            %     de           = obj(V*newp,params);
            %     %else
            %     %    improve = false;
            %     %end
            % 
            % end
        end
        
        % James-Stein estimator (shrinkage)
%         JSE  = 1 - (np-2)./sum(dx.^2);
%         ddx  = dx - x1;
%         dx   = x1 + JSE.*ddx; 
%         de   = obj(dx,params);
%         pupdate(loc,n,nfun,de,e1,'JSteinE',toc);
        
        
        % runge-kutta optimisation block: fine tune dx
        if rungekutta > 0
            pupdate(loc,n,nfun,de,e1,'RK lnsr',toc);
            
            %LB  = dx - exp(sqrt(red))*2;
            %UB  = dx + exp(sqrt(red))*2;
            
            % Make the U/L bounds proportional to the probability over the
            % prior variance
            LB  = denan( dx - ( pt(:)./red ) );
            UB  = denan( dx + ( pt(:)./red ) );
            
            B = find(UB==LB);
            
            LB(B) = dx(B) - 1;
            UB(B) = dx(B) + 1;
            
            %LB = LB.*(~~red);
            %UB = UB.*(~~red);
            
            %LB = dx - (dx.*red)*2;
            %UB = dx + (dx.*red)*2;
            
            SearchAgents_no = rungekutta;
            Max_iteration = rungekutta;
            
            dim = length(dx);
            fun = @(x) obj(x,params);
            
            try
                [Frk,rdx,~]=RUN(SearchAgents_no,Max_iteration,LB',UB',dim,fun,dx,red);
            
                rdx = rdx(:);
                dde = obj(rdx,params);

                if dde < de
                    dx = rdx(:);
                    de = dde;
                    
                end

                if verbose; pupdate(loc,n,nfun,de,e1,'RK fini',toc); end
                
                
            end
            
        end
        
        thisdist = cdist(dx',x1');
        if verbose; fprintf('| --> euc dist(dp) = %d\n',thisdist); end
        
        if thisdist < 1e-4
            break;
        end
        
        % update global points store
        all_dx = [all_dx dx(:)];
        all_ex = [all_ex de(:)];
        
        % % put polyfit optimisation here
        % if do_poly
        %     % fit a Gaussian Process Regression model to the available
        %     % points (this will have more points on each iteration)
        %     pupdate(loc,n,nfun,de,e1,'polyfit',toc);
        %     de  = obj(dx,params);
        % 
        %     % predictors (errors) and parameters
        %     if n > 1
        %         %ex = [Hist.e(:); de];
        %         %py = [cat(2,Hist.p{:}) dx]';
        %         ex = all_ex';
        %         py = all_dx';
        %     else
        %         ex = [e0 de]';
        %         py = [x1 dx]';
        %     end
        % 
        %     % model and prediction for each parameter
        %     for ipn = 1:length(x1)
        %         pf = polyfit(ex,py(:,ipn),2);
        %         yp(ipn) = polyval(pf,criterion);
        %     end
        % 
        %     if obj(yp(:),params) < de
        %         newe = obj(yp(:),params);
        %         pupdate(loc,n,nfun,e1,newe,'polpred',toc);
        %         dx = yp(:);
        %         de = newe;
        %     end
        % 
        % end
        % 
        % % put GP optimisation here
        % if do_gpr
        %     % fit a Gaussian Process Regression model to the available
        %     % points (this will have more points on each iteration)
        %     pupdate(loc,n,nfun,e1,e1,'GPR fit',toc);
        % 
        %     for dogpr = 1:do_gpr
        % 
        % 
        %         de  = obj(dx,params);
        % 
        %         % predictors (errors) and parameters
        %         if n > 1
        %             %ex = [Hist.e(:); de];
        %             %py = [cat(2,Hist.p{:}) dx]';
        %             ex = all_ex';
        %             py = all_dx';
        %         else
        %             ex = [e0 de]';
        %             py = [x1 dx]';
        %         end
        % 
        %         if isinf(criterion)
        %             critgp = de/2;
        %         else
        %             critgp = criterion;
        %         end
        % 
        %         % model and prediction for each parameter
        %         for ipn = 1:length(x1)
        % 
        %             gprm = fitrgp(ex,py(:,ipn),'Basis','constant','FitMethod','exact',...
        %             'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
        %             'Standardize',1,'Optimizer','fminsearch');
        %             [yp(ipn),se(ipn,:),yint(ipn,:)] = predict(gprm,critgp);
        %         end
        % 
        %         % compute objective at mean and both upper and lower
        %         % confidence intervals, use best
        %         sample_posterior = 0;
        %         if ~sample_posterior
        %             opts   = [yp(:) yint];
        %             emx(1) = obj(opts(:,1),params);
        %             emx(2) = obj(opts(:,2),params);
        %             emx(3) = obj(opts(:,3),params);
        %             [~,selI] = min(emx);
        %             yp = opts(:,selI);
        %         else
        %             % sample from the posterior distirbution of the GP
        %             opts   = [yp(:) yint];
        %             if n == 2
        %                 disp(' ');
        %             end
        %             for isamp = 1:length(dx)*4
        %                 %selc = randi([1 3],length(dx),1);
        %                 for selp = 1:length(dx)
        %                     %sam_dx(selp,isamp) = opts(selp,selc(selp));
        %                     r = yint(selp,1) + (yint(selp,2)-yint(selp,1)) .* rand(1,1);
        %                     sam_dx(selp,isamp) = r;
        % 
        %                 end
        %                 emx(isamp) = obj(sam_dx(:,isamp),params);
        %             end
        %             [~,selI] = min(emx);
        %             yp = sam_dx(:,selI);
        %         end
        % 
        %         % new predicted dx & variance
        %         %if emx(selI) < de
        %             pupdate(loc,n,nfun,e1,emx(selI),'GPRpred',toc);
        %             un  = yp(:) - dx(:);
        %             dx  = yp(:);
        %             de  = emx(selI);
        %             e1 = de;
        %         %end
        % 
        %         % save & return the GP prediction
        %         Hist.GP{n} = [yp(:) yint(:,1) yint(:,2)];
        %         Hist.GPi{n} = selI;
        %     end
        % end
                
        % Tolerance on update error as function of iteration number
        % - this can be helpful in functions with lots of local minima
        if givetol; 
                    %etol = e1 * ( ( 0.5./(n*2) )  );
                    etol = 1./1+exp(1./(n))/(maxit*2);
        else;       etol = 0; 
        end
        
        if etol ~= 0
            inner_loop=2;
        end
        
        deltap = cdist(dx',x1');
        deltaptol = 1e-6;
        
        
        % put memory optimisation here?
        
        % treat iterations as an optimisable quadratic integration scheme
        % i.e. dx_dot = dx + w([t-1])*history(dx[t-1]) + w([t-2]*h([t-2]) ...
        %
        % this effectively equips the optimisation with a 'memory' over
        % iteration cycles, with which to finess the gradient flow
        
        integration_nc=memory_optimise;
        if integration_nc && n == 1;
            %Hist.hyperdx(maxit+1,maxit) = nan;
            Hist.hyperdx= zeros(3,maxit);
        end
        
        if integration_nc && n > 1
            pupdate(loc,n,nfun,e1,e1,'mem int',toc);
            
            try
            
                % fminsearch memory-hyperparmeter options
                options.MaxIter = 25;
                options.Display = 'off';
                if doparallel
                    options.UseParallel = 1;
                end

                k  = [cat(2,Hist.p{:}) dx];

                % limit memory depth
                try; k = k(:,end-4:end); end

                hp = zeros(size(k,2),1);
                hp(end)=1;

                %k  = [XX0 x1 dx];
                %hp = [0 0 1]';

                % memory [k] and weights [x]
                gx = @(x) obj(k*x,params);
                %X  = fminsearch(gx,hp,options);

                
                LB = zeros(size(hp))-1;
                UB = ones(size(hp))+2;
                dim = length(hp);

                SearchAgents_no = 6;
                Max_iteration = 6*2;

                fun = @(x) obj(k*x(:),params);

                [Frk,X,~]=RUN(SearchAgents_no,Max_iteration,LB',UB',dim,fun,hp,ones(size(hp))/8);              
                X=X(:);

                %X = fmincon(gx,hp,[],[],[],[],0*hp,2*ones(size(hp)),[],options);

                %J  = jaco(gx,hp,ones(size(hp))*1e-3,0,2);
                %dx = k*(hp - (hp.*J));
                %X  = J;

                %X = fminunc(gx,hp,options);
                %X = X./sum(X);
                
                if obj(k*X,params) < obj(dx,params)
                    dx = k*X;
                    de = obj(dx,params);
                    if verbose; fprintf('Memory helped improve gradient flow update\n');end
                end

                try
                    if n < 3
                       Hist.hyperdx(:,n) = Hist.hyperdx(:,n) + [X];
                    else
                        Hist.hyperdx(:,n) = Hist.hyperdx(:,n) + X(end-2:end);
                    end
                catch 
                    try
                        Hist.hyperdx(:,n) = [X];
                    catch
                        Hist.hyperdx(:,n) = 0;
                    end
                end

                %Hist.hyperdx(1:size(k,2),n) = X;

                % plot
                s(1) = subplot(5,3,11);cla;
                %surf(Hist.hyperdx'); shading interp;
                %hold on;
                block = Hist.hyperdx;
                %block(3,:) = block(3,:) * .001;
                imagesc(block);
                caxis([-1 1]*.01);
                colormap(cmocean('balance'));
                %grid off;
                title('Update Rate','color','w','fontsize',18);
                %xlabel('Iteration','color','w');
                s(1).YColor = [1 1 1];
                s(1).XColor = [1 1 1];
                s(1).Color  = [.3 .3 .3];
                set(gca,'ytick',1:3,'yticklabel',{'Mem1' 'Mem2' 'GradFlow'});
                
            end
            
            %ylim([-.2 1.2]);
            %view([-38.4169   78.1306]);
            %legend({'Wt: Baseline' 'Wt: Memory', 'Wt: Gradient Flow'},'color','w','location','southoutside');
            %legend({'Gradient Flow' 'Memories'},'color','w','location','southoutside');
            
        end
                
        % log deltaxs
        Hist.dx(:,n) = dx;
        
        % if predictionerrorupdate
        %    % If we ignore nonlinearities between parameter effects, we can
        %    % propogate the prediction error associated with each parameters update
        %    % to improve the error / prediction
        %    fprintf('\nUpdating Prediction Errors...\n');
        % 
        %    ee = obj(x1,params);
        % 
        %    for ik = 1:length(x1)
        %       ddx     = x1;
        %       ddx(ik) = dx(ik);
        %       ex(ik)  = obj(ddx,params);
        %    end
        % 
        %    delta_y = (ex - ee)./ee;
        %    delta_p = (dx - x1)./x1;
        % 
        %    dydp = delta_y(:) ./ delta_p(:);
        % 
        %    % predicted objective under linear assumption
        %    predic = sum( delta_y(:).*(dx-dydp) );
        % 
        %    fprintf('Prediction = %d --> ',e0);           
        % 
        %    % Y(i) = b0 + b1*X(i) + e(i)
        %    pe_dx = dx - dydp*1e-4;
        % 
        %    pe_e = obj(pe_dx,params);
        % 
        %    % actual objective
        %    fprintf('Error-Update = %d',pe_e);
        % 
        %    if pe_e < obj(dx,params)
        %        de = pe_e;
        %        dx = pe_dx;
        %        fprintf(' [accept]\n\n');
        %    else
        %        fprintf(' [ignore]\n\n');
        %    end
        % 
        % end

        % print prediction
        pupdate(loc,n,nfun,de,e1,'predict',toc);
        
        
        % Evaluation of the prediction(s)
        %------------------------------------------------------------------
        if de  < ( obj(x1,params) + abs(etol) ) && (deltap > deltaptol)
            
            % If the objective function has improved...
            if verbose; if nfun == 1; pupdate(loc,n,nfun,de,e1,'improve',toc); end; end
            
            % update the error & the (reduced) parameter set
            %--------------------------------------------------------------
            df  = e1 - de;
            e1  = de;
            x1  = V'*dx;
            
            aopt.modpred(:,n) = spm_vec(params.aopt.fun(spm_unvec(x1,aopt.x0x0)));
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
        
        % increase learning rate
        %red = red * 1.1;
                
        % Extrapolate...
        %==================================================================        
        
        % we know what param-step caused what improvement, so try again...
        %------------------------------------------------------------------
        exploit = true;
        nexpl   = 0;
        if verbose; pupdate(loc,n,nexpl,e1,e0,'descend',toc);end
                
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
                if verbose;pupdate(loc,n,nexpl,e1,e0,'finish ',toc);end
            end
            
            % upper limit on the length of this loop: no don't do this
            if nexpl == (inner_loop)
                exploit = false;
            end
        end
        
        % Update best-so-far estimates
        e0 = e1;
        x0 = x1;
        
        % store mode prediction
        aopt.modpred(:,n) = spm_vec(params.aopt.fun(spm_unvec(x0,aopt.x0x0)));    
        
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
        pupdate(loc,n,nfun,e1,e0,'linsrch',toc);    
        e_orig = e0;
                
        % select only parameters whose steps improved the objective
        %------------------------------------------------------------------
        thisgood = gp*0; % track whether any of selection get used
        if any(gp)
            
            % sort good params by (improvement amount) * (probability)
            %--------------------------------------------------------------
            % update p's causing biggest improvment in fe while maintaining highest P(p)
            [~,PO] = sort(rescale(real(-DFE(gpi))),'descend');
            
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
                        aopt.modpred(:,n) = spm_vec(params.aopt.fun(spm_unvec(x0,aopt.x0x0)));  
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
                    %red = red*.8;
                    
                    warning off;try df; catch df = 0; end;warning on;
                    
                    % OK, 2022: do this on every fail
                    %-------------------------------
                    %red     = diag(pC);
                    %aopt.pC = V*red;
                    
                    % Halt this while loop % Keep counting rejections
                    improve1 = 0;
                    n_reject_consec = n_reject_consec + 1;
                    
                    % decrease learning rate by 50%
                    red = red * .5;
                    
                end
                
                % update global store of V
                aopt.pC = V*red;
            end
            
        else
            if sample_mvn
                % sample from a multivariate normal over parameter set
                pupdate(loc,n,nfun,e0,e0,'sample ',toc);
                xnew = x0;
                for ilp = 1:length(x1)

                    %R    = x0 - mvnrnd(x0,(aopt.Cp'+aopt.Cp)/2,1)';
                    %xnew = xnew + R(:);
                    xnew = x0 - (red.^2).*(8*mvnrnd(x0,(aopt.Cp'+aopt.Cp)/2,1))';
                    enew = obj(xnew,params);

                    if enew < (e0+abs(etol))
                        x0  = V'*(xnew);
                        df  = enew - e_orig;
                        e0  = enew;

                        % Print & plot update
                        nupdate = [length(find(x0 - aopt.pp)) length(x0)];
                        pupdate(loc,n,nfun,e0,e0,'accept ',toc,nupdate);
                    else
                        df = 0;
                    end
                end

            else
            
            
                % If we get here, then there were not gradient steps across any
                % parameters which improved the objective! So...
                pupdate(loc,n,nfun,e0,e0,'reject ',toc);

                % Our (prior) 'variances' are probably too small
                %red = red*1.4;
                
                % decrease learning rate by 50%
                red = red * .5;
                
                % OK, 2022: do this on every fail
                %-------------------------------
                %red     = diag(pC);
                %aopt.pC = V*red;                

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
        
    end
    
    % Stopping criteria, rules etc.
    %======================================================================
    crit = [ (abs(df(1)) < min_df) crit(1:end - 1) ]; 
    clear df;
    
    if all(crit)
        localminflag = 3;            
    end
        
    if localminflag == 3
        fprintf(loc,'We''re either stuck or converged...stopping.\n');
        [X,F,Cp,PP] = finishup(V,x0,ip,e0,doparallel,params,J,Ep,red,writelog,loc,aopt);
        return;
        
    end
    
    % If 3 fails, reset the reduction term (based on the specified variance)
    if n_reject_consec == 2
        pupdate(loc,n,nfun,e1,e0,'resetv');
        red     = diag(pC);
        aopt.pC = V*red;
        if n == 1 && search_method~=100 ; a = red*0;
        end
        
    end
    
    % stop at max iterations
    if n == maxit
        fprintf(loc,'Reached maximum iterations: stopping.\n');
        [X,F,Cp,PP] = finishup(V,x0,ip,e0,doparallel,params,J,Ep,red,writelog,loc,aopt);
        return;
    end
    
    % check for convergence
    if e0 <= criterion
        fprintf(loc,'Convergence.\n');
        [X,F,Cp,PP] = finishup(V,x0,ip,e0,doparallel,params,J,Ep,red,writelog,loc,aopt);
        return;
    end
    
    % give up after 10 failed iterations
    if n_reject_consec == 6
        fprintf(loc,'Failed to converge...\n');
        [X,F,Cp,PP] = finishup(V,x0,ip,e0,doparallel,params,J,Ep,red,writelog,loc,aopt);
        return;
     end
end
    

end

% Subfunctions: Plotting & printing updates to the console / log...
%==========================================================================

% function [X,F,Cp,PP] = userstop(x,y,varargin)
% 
% if strcmp(y.Character,'c')
%     disp('User stop initiated');
% 
%     % send it back to the caller (AO.m)
%     C = {'V','x0','ip','e0','doparallel','params','J','Ep','red','writelog','loc','aopt'};
% 
%     
% %st = dbstack('-completenames');
% %    this = find(strcmp('AO',{st.name}));
%         
%     
%     [X,F,Cp,PP] = finishup(V,x0,ip,e0,doparallel,params,J,Ep,red,writelog,loc,aopt);
%     return;
%     
% end
% end

function [X,F,Cp,PP] = finishup(V,x0,ip,e0,doparallel,params,J,Ep,red,writelog,loc,aopt);

fprintf(loc,'Finishing up...\n');

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
        
end

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

s = subplot(5,3,3);
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
        

function f = setfig()
% Main figure initiation

%figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off','position',[1088 122 442 914]);
%figpos = get(0,'defaultfigureposition').*[1 1 0 0] + [0 0 710 842];
%figpos = get(0,'defaultfigureposition').*[1 1 0 0] + [0 0 910 842];
figpos = get(0,'defaultfigureposition');

%1          87        1024        1730

%figpos = [816         405        1082        1134];
f = figure('Name','AO Optimiser','Color',[.3 .3 .3],'InvertHardcopy','off','position',figpos); % [2436,360,710,842]
set(gcf, 'MenuBar', 'none');
set(gcf, 'ToolBar', 'none');
drawnow;
set(0,'DefaultAxesTitleFontWeight','normal');
end

function BayesPlot(x,pr,po)
% Update the Bayes plot

s(3) = subplot(5,2,5);
imagesc(pr);
s(4) = subplot(5,2,6);
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

function plot_h_opt(x)

s(1) = subplot(5,3,13);

try
    plot(spm_vec(x(:,end-1)),'w:','linewidth',3); hold on;
end

plot(spm_vec(x(:,end)),'linewidth',3,'Color',[1 .7 .7]);

grid on;
title('Precision Hyperprm','color','w','fontsize',18);hold off;

s(1).YColor = [1 1 1];
s(1).XColor = [1 1 1];
s(1).Color  = [.3 .3 .3];    

end

function plot_hyper(x,y)

% if hypertune; plot_hyper(params.hyper_tau); end

s(1) = subplot(5,3,12);

plot3(1:length(x),real(spm_vec(x)),real(y),'Color',[1 .7 .7],'linewidth',3); hold on;
scatter3(1:length(x),real(spm_vec(x)),real(y),30,'w','filled');
grid on;
title('Exp Hypr Tune','color','w','fontsize',18);
xlabel('Iteration','color','w');
ylabel('h','color','w');
zlabel('e','color','w');
s(1).YColor = [1 1 1];
s(1).XColor = [1 1 1];
s(1).ZColor = [1 1 1];
s(1).Color  = [.3 .3 .3];
grid minor;


end

function pl_init(x,params)
% A subplot in the main figure of the start point of the algorithm
[Y,y] = GetStates(x,params);

% Restrict plots to real values only - just for clarity really
% (this doesn't mean the actual model output is not complex)
Y = spm_unvec( real(spm_vec(Y)), Y);
y = spm_unvec( real(spm_vec(y)), Y);

s(1) = subplot(5,3,[10]);

plot(spm_vec(Y),'w:','linewidth',3); hold on;
plot(spm_vec(y),     'linewidth',3,'Color',[1 .7 .7]); hold off;
grid on;grid minor;title('Start Point','color','w','fontsize',18);
s(1).YColor = [1 1 1];
s(1).XColor = [1 1 1];
s(1).Color  = [.3 .3 .3];

end

function prplot(pt)
% Updating probability subplot

s = subplot(5,3,8);
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
    s = subplot(5,3,[9]);
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
    subplot(5,3,15);hold off;
    feval(params.userplotfun,x,params);
    ax        = gca;
    ax.YColor = [1 1 1];
    ax.XColor = [1 1 1];
    ax.Color  = [.3 .3 .3];
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
        s(1) = subplot(5,3,[1 2]);

        plot(spm_vec(Y),'w:','linewidth',3); hold on;
        plot(spm_vec(y),     'linewidth',3,'Color',[1 .7 .7]); hold off;
        grid on;grid minor;title('AO MAP Estimation: Current Best','color','w','fontsize',18);
        s(1).YColor = [1 1 1];
        s(1).XColor = [1 1 1];
        s(1).Color  = [.3 .3 .3];
    else
        s(1) = subplot(5,3,1);
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
        s(6) = subplot(5,3,2);
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
    s(2) = subplot(5,3,[6]);
    %bar([former_error new_error]);
    plot(former_error,'w--','linewidth',3); hold on;
    plot(new_error,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;title('Error Change','color','w','fontsize',18);
    ylabel('Δ error');
    s(2).YColor = [1 1 1];
    s(2).XColor = [1 1 1];
    s(2).Color  = [.3 .3 .3];
    
    
    %s(3) = subplot(413);
    s(3) = subplot(5,3,7);
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
    
    s(4) = subplot(5,3,[4 5]);
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
    
    if isfield(params,'r2')
        params.r2 = [params.r2(:); corr(spm_vec(Y),spm_vec(y)).^2 ];
    else
        params.r2 = corr(spm_vec(Y),spm_vec(y)).^2;
    end
    
    s(5) = subplot(5,3,9);
    plot(spm_vec(params.r2),'w:','linewidth',3);
    %hold on;
    %plot(spm_vec(Y)-aopt.oerror,'linewidth',3,'Color',[1 .7 .7]); hold off;
    grid on;grid minor;
    til = sprintf('Variance Expln: %d%%',100*round(params.r2(end)*1000)/1000);
    title(til,'color','w','fontsize',18);
    ax = gca;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    s(5).YColor = [1 1 1];
    s(5).XColor = [1 1 1];
    s(5).Color  = [.3 .3 .3];
    ylim([0 1]);
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

function [e,J,er,mp,Cp,L,params] = obj(x0,params,varargin)
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

% More chekcks - attempt to catch the case where optimisation has pushed the
% parameters such that the system is unstable and returning empty or nan or
% inf etc. 
y = denan(y);
if length(y) == 0
    y = Y*0 + inf;
end

% Feature selection
%--------------------------------------------------------------------------
if isfield(params,'FS')
    yfs = params.FS(spm_unvec(y,Y));
    Yfs = params.FS(Y);
    
    y = spm_vec(yfs);
    Y = spm_vec(Yfs);
    
    if isnumeric(Q) && ~isempty(Q)
        n1 = length(y) - length(Q);
        if n1 > 0
            Q(end+1:end+n1,end+1:end+n1) = eye(n1); % pad out Q
        elseif n1 < 0
            Q = eye(length(yfs));
        end
    end
end

% if all(y) = 0 then the feature selection may fail, so double check:
if length(Y) > length(y)
    y(end+1:length(Y)) = 0;
end

% ensure vectors!
Y = Y(:);
y = y(:);

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
    clear Q;
    lpq = length(aopt.precisionQ);
    for ijq = 1:length(aopt.precisionQ)
       Q{ijq} = sparse(ijq,ijq,aopt.precisionQ(ijq,ijq),lpq,lpq);
    end

    ny  = length(spm_vec(y));
    nq  = ny ./ length(Q{1});
end

% if nargin > 2
%     hh = varargin{1};
% else
%     try   hh = aopt.hh;
%     catch hh = 1;
%     end
% end


if ~isfield(aopt,'h') || ~aopt.hyperparameters
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

%iS = iS + Q.*(exp(-32) + exp(h));

%h

e   = (spm_vec(Y) - spm_vec(y)).^2;

%S = atcm.fun.QtoGauss(Y,12) - atcm.fun.QtoGauss(y,12);
%e = max(S')';

%e   = (spm_vec(Y) - spm_vec(y));
ipC = aopt.ipC;

warning off;                                % suppress singularity warnings
Cp  = spm_inv( (aopt.J*iS*aopt.J') + ipC );
%Cp = (Cp + Cp')./2;
warning on

% if aopt.rankappropriate
%     N = rank((Cp)); % cov rank
%     [v,D] = eig(real(mean(e)) + (Cp)); % decompose covariance matrix
%     DD  = diag(D); [~,ord]=sort(DD,'descend'); % sort eigenvalues
%     Cp = v(:,ord(1:N))*D(ord(1:N),ord(1:N))*v(:,ord(1:N))';
% end

p  = ( x0(:) - aopt.pp(:) );

if any(isnan(Cp(:))) 
    Cp = Cp;
end


if aopt.hyperparameters
%     if isfield(params.aopt,'h') && ~isempty(params.aopt.h) && params.aopt.computeh
%         %fprintf('using precomputed hyperparam while differentiating\n');
%         h = params.aopt.h;
%         ihC = params.aopt.ihC;
%         d = params.aopt.d;
%         Ch = params.aopt.Ch;
%     else
        %fprintf('\n\n\ncomputing h\n\n\n');
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
                JPJ{i} = real(aopt.J*P{i}*aopt.J');
            end
%                 i=1;
%                 P{i}   = Q.*exp(h);
%                 PS{i}  = P{i}*S;
%                 %P{i}   = kron(speye(nq),P{i});
%                 JPJ{i} = real(aopt.J*P{i}*aopt.J');
         

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

%             for i = 1:nh
%                 dFdh(i,1)      =   trace(diag(PS{1}(:,i)))*nq/2 ...
%                     - real(e'*diag(P{1}(:,i))*e)/2 ...
%                     - spm_trace(Cp,diag(JPJ{1}(:,1)))/2;
%                 for j = i:nh
%                     dFdhh(i,j) = - spm_trace(diag(PS{1}(:,i)),diag(PS{1}(:,j)))*nq/2;
%                     dFdhh(j,i) =   dFdhh(i,j);
%                 end
%             end


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
      %  end
  %  end
end % end of if hyperparams (from spm) ... 

% record hyperparameter h over iterations
if aopt.hyperparameters && aopt.updateh
    if isfield(params,'h_opt') && ~isempty(params.h_opt) 
        params.h_opt = [params.h_opt h];
    else
        params.h_opt = h;
    end
end

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
            e  = ( (norm(full(er),2).^2)/numel(spm_vec(Y)) ).^(1/2);

            
        case 'mvgkl'
            % multivariate gaussian kullback lieb div
            
            %covQ = aopt.Q;
            %covQ(covQ<0)=-covQ(covQ<0);
            %covQ = (covQ + covQ')/2;
            
            % pad for when using FS(y) ~= length(y)
            %padv = length(Y) - length(covQ);
            %covQ(end+1:end+padv,end+1:end+padv)=.1;
            
            % make sure its positive semidefinite
            %lbdmin = min(eig(covQ));
            %boost = 2;
            %covQ = covQ + ( boost * max(-lbdmin,0)*eye(size(covQ)) );
            
            cY = atcm.fun.VtoGauss(real(Y));
            cy = atcm.fun.VtoGauss(real(y));


            % truth [Y] first = i.e. inclusive, mean-seeking
            % https://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/
            e = mvgkl(Y,cY,spm_vec(y),cy);
                        
        case 'q'
                                 
            er = (AGenQ(Y)-AGenQ(y));
            e  = ( (norm(full(er),2).^2)/numel(spm_vec(Y)) ).^(1/2);
                        
        case 'gaussfe'

            % Gaussian erorr term using Frobenius distance
            dgY = atcm.fun.VtoGauss(real(Y));
            dgy = atcm.fun.VtoGauss(real(y));
            
            Dg  = dgY - dgy;
            e   = diag(Dg*Dg');

            L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2;

            % (2) Complexity minus accuracy of parameters
            %--------------------------------------------------------------------------
            L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;

            if aopt.hyperparameters
                % (3) Complexity minus accuracy of precision
                %----------------------------------------------------------------------
                L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;
            end
    
            F    = sum(L);
            e    = (-F);
            

        case 'gaussmap'

            % Gaussian erorr term using Frobenius distance
            dgY = atcm.fun.VtoGauss(real(Y),12*2);
            dgy = atcm.fun.VtoGauss(real(y),12*2);
            
            Dg  = dgY - dgy;
            e   = trace(Dg*Dg');

            if aopt.hyperparameters
                e = e - spm_logdet(ihC*Ch)/2 - d'*ihC*d/2; 
            end

            % Parameter p(th) given (prior) distributions
            for i = 1:length(p)
                vv     = real(sqrt( Cp(i,i) ))*2;
                if vv <= 0 || isnan(vv) || isinf(vv); vv = 1/64; end
                pd(i)  = makedist('normal','mu', real(aopt.pp(i)),'sigma', vv);
                pdx(i) = normcdf(x0(i),pd(i).mu,pd(i).sigma);
            end

            % full map: log(f(X|p)) + log(g(p))
            e         = log(e) + 1./(1-log(prod(pdx*2)));


        case {'gauss' 'gp'}
            
            % first  pass gauss error
            dgY = atcm.fun.QtoGauss(real(Y),12*2);
            dgy = atcm.fun.QtoGauss(real(y),12*2);
            Dg  = dgY - dgy;
            e   = trace(Dg'*Dg);
           
            
            if aopt.hyperparameters
                % (3) Complexity minus accuracy of precision
                %----------------------------------------------------------------------
                e = e - spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;     
                %L(2) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;;
            end

        case 'gaussv'

            % first  pass gauss error
            dgY = atcm.fun.VtoGauss(real(Y));
            dgy = atcm.fun.VtoGauss(real(y));
            Dg  = dgY - dgy;
            e   = trace(Dg'*Dg);
           
            
            if aopt.hyperparameters
                % (3) Complexity minus accuracy of precision
                %----------------------------------------------------------------------
                e = e - spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;     
                %L(2) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;;
            end

        case 'gauss_components'

            [dgY] = atcm.fun.approxlinfitgaussian(Y);
            [dgy] = atcm.fun.approxlinfitgaussian(y);

            Dg  = cdist(dgY,dgy);
            e   = trace(Dg'*Dg);



    case {'gausspowspec'}
        % a slightly extended version of the gauss error function but with 
            
            % first  pass gauss error
            widths = [];
            dgY = atcm.fun.VtoGauss(real(Y));
            dgy = atcm.fun.VtoGauss(real(y));
            Dg  = dgY - dgy;
            e   = trace(Dg'*Dg);

            % indices of biggest to smallest points ...
            XY = atcm.fun.maxpointsinds(Y,length(Y));
            Xy = atcm.fun.maxpointsinds(y,length(y));
            
            % difference in position for each element
            YIND = XY*0;
            for i = 1:length(Y)                
               YIND(XY(i)) = find(XY(i)==Xy);
            end

            % place index difference into error
            Dg = Dg*diag(YIND)*Dg';
            e  = trace(Dg'*Dg);
           
            
            if aopt.hyperparameters
                % (3) Complexity minus accuracy of precision
                %----------------------------------------------------------------------
                e = e - spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;     
                %L(2) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;;
            end
            
        case {'gaussnorm'}
            
            % first  pass gauss error
            dgY = atcm.fun.QtoGauss(real(Y),12*2);
            dgy = atcm.fun.QtoGauss(real(y),12*2);
            Dg  = dgY - dgy;
            e   = norm(Dg'*Dg);
           
            
            if aopt.hyperparameters
                % (3) Complexity minus accuracy of precision
                %----------------------------------------------------------------------
                e = e - spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;     
                %L(2) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;;
            end


        % case 'gaussv'
        % 
        %     % first  pass gauss error
        %     dgY = atcm.fun.VtoGauss(real(Y));
        %     dgy = atcm.fun.VtoGauss(real(y));
        %     Dg  = dgY - dgy;
        %     e   = trace(Dg'*Dg);
        % 
        % 
        %     if aopt.hyperparameters
        %         % (3) Complexity minus accuracy of precision
        %         %----------------------------------------------------------------------
        %         e = e - spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;     
        %         %L(2) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;;
        %     end
            
            %F    = sum(L);
            %e    = real(-F);
        case 'gausscluster'

            dY = atcm.fun.clustervec(Y);
            dy = atcm.fun.clustervec(y);

            Dg = cdist(dY,dy);
            e  = sum(min(Dg)) + sum(min(Dg'));

        case 'distancewei'

            dY = distance_wei(fast_HVG(Y,1:length(Y)));
            dy = distance_wei(fast_HVG(y,1:length(Y)));

            Dg = dY - dy;

            e = trace(Dg*Dg');

        case 'gaussq'
                        
            % first  pass gauss error
            %dgY = atcm.fun.QtoGauss(real(Y),12*2);
            %dgy = atcm.fun.QtoGauss(real(y),12*2);
            
            [dgY,~,qY] = atcm.fun.gausvdpca(real(Y));
            [dgy,~,qy] = atcm.fun.gausvdpca(real(y));

            Dg  = dgY - dgy;
            
            e   = trace(Dg*Dg');

            %parameters
%             try
%                 dxpt = aopt.pt(:,end);
%                 xpt  = aopt.pt(:,end-1);
% 
%                 ep = atcm.fun.gausvdpca(xpt) - atcm.fun.gausvdpca(dxpt);
% 
%                 e = log(e) + log( trace(ep*ep') );
% 
%             end

            
            
%             if isfield(aopt,'precisionQ')
%                 pQ = aopt.precisionQ; % use pQ not Q otherwise it becomes stochastic!
%                 padv = length(Y) - length(pQ);
%                 pQ(end+1:end+padv,end+1:end+padv)=mean(pQ(:))/10;      
%                 pQ = gaufun.GaussPCA(pQ,20);
%             end
%          
%             % first  pass gauss error
%             dgY = atcm.fun.QtoGauss(real(Y),12*2);
%             dgy = atcm.fun.QtoGauss(real(y),12*2);
%             Dg  = dgY - dgy;
%             K   = rank(Dg) + 1;
%             
%             % second pass with rank
%             dgY = atcm.fun.QtoGauss(real(Y),K);
%             dgy = atcm.fun.QtoGauss(real(y),K);
%             Dg  = dgY - dgy;
% 
%             e  = trace(Dg*pQ*Dg');

            
            if aopt.hyperparameters
                % (3) Complexity minus accuracy of precision
                %----------------------------------------------------------------------
                e = e - spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;     
                %L(2) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;;
            end
            
            %F    = sum(L);
            %e    = real(-F);
                        
        case 'gaussfe'
            dY = atcm.fun.QtoGauss(real(Y),12*2);
            dy = atcm.fun.QtoGauss(real(y),12*2);

            D = dY - dy;
            e = diag(D*D');

            L(1) = spm_logdet(iS)*nq/2  - real(e'*iS*e)/2 - ny*log(8*atan(1))/2; 
            L(2) = spm_logdet(ipC*Cp)/2 - p'*ipC*p/2;
            L(3) = spm_logdet(ihC*Ch)/2 - d'*ihC*d/2;
            F    = sum(L);
            e    = real(-F);
            
        case 'jsd'
            % Jensen-SHannon divergence using multivariate gaussian kullback lieb div
            
            covQ = aopt.Q;
            covQ(covQ<0)=0;
            covQ = (covQ + covQ')/2;
            
            % pad for when using FS(y) ~= length(y)
            padv = length(Y) - length(covQ);
            covQ(end+1:end+padv,end+1:end+padv)=.1;
            
            % make sure its positive semidefinite
            lbdmin = min(eig(covQ));
            boost = 2;
            covQ = covQ + ( boost * max(-lbdmin,0)*eye(size(covQ)) );
            
            % truth [Y] first = i.e. inclusive, mean-seeking
            % https://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/
            e = 0.5*mvgkl(Y,covQ,y(:),covQ) + 0.5*mvgkl(y(:),covQ,Y,covQ);
            
            e = abs(e);
                        
        case 'mvgkl_rmse'
            % multivariate gaussian kullback lieb div
            
            covQ = aopt.Q;
            covQ(covQ<0)=0;
            covQ = (covQ + covQ')/2;
            
            % pad for when using FS(y) ~= length(y)
            padv = length(Y) - length(covQ);
            covQ(end+1:end+padv,end+1:end+padv)=.1;
            
            % make sure its positive semidefinite
            lbdmin = min(eig(covQ));
            boost = 2;
            covQ = covQ + ( boost * max(-lbdmin,0)*eye(size(covQ)) );
            
            % truth [Y] first = i.e. inclusive, mean-seeking
            % https://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/
            e = mvgkl(Y,covQ,y(:),covQ);
            
            %er = spm_vec(Y)-spm_vec(y);
            %e  = e * ( (norm(full(er),2).^2)/numel(spm_vec(Y)) ).^(1/2);
            
            er = (spm_vec(Y)-spm_vec(y)).^2;
            ed = cdist(Y,y);
            
            e = e + ( er'*ed*er )/2; 
            
            
        case 'mvgklx'
            % multivariate gaussian kullback lieb div - minimise the
            % divergence between the model and data, as well as the propoability of the
            % params
            
            covQ = aopt.Q;
            covQ(covQ<0)=0;
            covQ = (covQ + covQ')/2;
            
            % pad for when using FS(y) ~= length(y)
            padv = length(Y) - length(covQ);
            covQ(end+1:end+padv,end+1:end+padv)=.1;
            
            % make sure its positive semidefinite
            lbdmin = min(eig(covQ));
            boost = 2;
            covQ = covQ + ( boost * max(-lbdmin,0)*eye(size(covQ)) );
            
            
            e = mvgkl(Y,covQ,y(:),covQ);
            
            % KL(Data|Model) + KL( p(dx)|p(x) )
            if size(aopt.pt,2) == 1
                dxpt = aopt.pt(:,end);
                dxpt = dxpt*dxpt';
                e = e + log( mvgkl(P(:),makeposdef(dxpt),aopt.pp(:),makeposdef(dxpt)) );
            
            else
                dxpt = aopt.pt(:,end);
                xpt  = aopt.pt(:,end-1);
                e = e + log( mvgkl(P(:),makeposdef(dxpt*dxpt'),aopt.pp(:),makeposdef(xpt*xpt')) );
            end
                
                
            
                    
        case 'mahal'    
            
            e = mahal(Y,y);
            e = (e'*iS*e)/2;
                        
        case 'lognorm'
            
            e=length(Y)/2*log(norm(Y-y));
            
        case {'qrmse' 'q_rmse'}
            % rmse: root mean squaree error incorporating precision
            % components
            er = spm_vec(Y)-spm_vec(y);
            
            % which Q ?
            if aopt.hyperparameters
                er = real(er'.*iS.*er)/2;
            else
                er = real(er'.*aopt.precisionQ.*er)/2;
            end
            
            er = full(er);
            e  = ( (norm(er,2).^2)/numel(spm_vec(Y)) ).^(1/2);
                       
            
        case {'correlation','corr','cor','r2'}
            % 1 - r^2 (bc. minimisation routine == maximisation)
            e = 1 - ( distcorr( spm_vec(Y), spm_vec(y) ).^2 );
            e = abs(e) .* abs(1 - (Y(:)'*y(:)./sum(Y.^2)));
            

        case 'combination'
            % combination:
            SSE = sum( ((spm_vec(Y) - spm_vec(y)).^2)./sum(spm_vec(Y)) );
            R2  = 1 - abs( corr( spm_vec(Y), spm_vec(y) ).^2 );
            e   = SSE + R2;
            
        case 'euclidean'
        
            ED = cdist(spm_vec(Y),spm_vec(y));
            ED = ED*iS*ED';
            e  = sum(spm_vec(ED)).^2;
            
        case {'rmse_euc' 'bregman'}
            % rmse: root mean squaree error incorporating precision
            % components
            er = spm_vec(Y)-spm_vec(y);
            er = real(er'.*iS.*er)/2;
            er = full(er);
            dv = cdist(Y,y);
            er = (er.*dv);
            e  = ( (norm(er,2).^2)/numel(spm_vec(Y)) ).^(1/2);
            
                         
        case {'g_kld' 'gkld' 'gkl' 'generalised_kld'}
             e = sum( denan(Y.*log(Y./y)) ) - sum(Y) - sum(y);
             
        case {'kl' 'kldiv' 'divergence'}';
            temp = denan(Y.*log(Y./y));
            temp(isnan(temp))=0;
            e   = sum(temp(:));

         case {'itakura-saito' 'is' 'isd'};
             e = sum( (Y./y) - denan(log(Y./y)) - 1 );
        
        case 'hvg_gl'
        
            w  = (1:length(Y))';
            Q  = fast_HVG(Y,w);
            A  = Q .* ~eye(length(Q));
            N  = size(A,1);
            GLY = speye(N,N) + (A - spdiags(sum(A,2),0,N,N))/4;
            
            Q   = fast_HVG(y,w);
            A   = Q .* ~eye(length(Q));
            N   = size(A,1);
            GLy = speye(N,N) + (A - spdiags(sum(A,2),0,N,N))/4;
            
            % frobenius distance
            e = sqrt( trace((GLY-GLy)*(GLY-GLy)') );
            
            %e = full(sum( (GLY(:)-GLy(:)).^2 ));
            
             
        case 'mle';
                        
            % we can perform parameter estimation by maximum likelihood estimation 
            % by minimising the negative log likelihood
            warning off;
                       
            w  = (1:length(Y)).';
            Y0 =  Y;% fit(w,Y,'Gauss4');
            y0 = y;% fit(w,y,'Gauss4');
            %e = log(sum(Y0(w)-y0(w)));
            e = fitgmdist(Y0-y0,2);
            e = e.NegativeLogLikelihood;
            warning on;

        case {'logistic' 'lr'}
            % logistic optimisation 
            e = -( spm_vec(Y)'*log(spm_vec(y)) + (1 - spm_vec(Y))'*log(1-spm_vec(y)) );
end


% hyperparameter [noise] tuning by GD - by placing this here it can easily
% be applied to any of the above cost functions
%--------------------------------------------------------------------------
if aopt.hypertune
    if isfield(params,'hyper_tau') && length(params.hyper_tau) > 1
        t = params.hyper_tau(end);
    else
        t  = .5;
    end
    
    %t  = .5;
    fc = @(t) t*exp( (1./t)*e );
    t  = hypertune(fc,t);
    de = fc(t);

    if de < e && (real(de)~= -inf);
        e = de;
    else
        
    end
    
    % return the tuned parameter for plotting
    try;
        params.hyper_tau = [params.hyper_tau; t];
    catch
        params.hyper_tau = t;
    end
end


%if aopt.hyperparameters
%    e = e - L(3);
%end

% if aopt.hypertune && length(params.hyper_tau)>1
%     if fc(params.hyper_tau(end-1)) < fc(t)
%         params.hyper_tau(end) = params.hyper_tau(end-1);
%         e = fc(params.hyper_tau(end));
%         fprintf('Retaining prious hyperparam\n');
%     end
% end

% Error along output vector (when fun is a vector output function)
er = e*( spm_vec(y) - spm_vec(Y) );
mp = spm_vec(y);


function t = hypertune(fc,t,N)
    if nargin<3; N=1000;end

    % hyperparam tuning by descent on hyperparameter
    jc = sign((fc(t) - fc(t + 1e-3)) ./ 1e-3);
    dt = t + jc*1e-3;
    nt = 0;
    
    while fc(dt) < fc(t) && nt < N
        nt = nt + 1;
        t  = dt;
        jc = sign((fc(t) - fc(t + 1e-3)) ./ 1e-3);
        dt = t + jc*1e-3;
    end
end

function t = hypertuner(fc,t,N)
    if nargin<3; N=1000;end
    % hyperparam tuning by descent on hyperparameter
    jc = sign((fc(t) - fc(t + 1e-3)) ./ 1e-3);
    dt = t + jc*1e-3;
    nt = 0;
    
    while fc(dt) < fc(t) && nt < N
        nt = nt + 1;
        t  = dt;
        jc = sign((fc(t) - fc(t + 1e-3)) ./ 1e-3);
        dt = t + jc*1e-3;
    end
end
        
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
    V(isnan(V))=0;
    Ord  = aopt.order; 
    
    % Switch for different steps in numerical differentiation
    if aopt.fixedstepderiv == 1
        V = (~~V)*exp(-8);
    elseif aopt.fixedstepderiv == -1
        nds = max(1e-7*ones(1,length(x0)),abs(x0(:)')*1e-4);
        V = nds(:).*(~~V);
        %V = V + ( (~~V) .* abs(randn(size(V))) );
    elseif aopt.fixedstepderiv == 0
        V = V;
    else %aopt.fixedstepderiv 
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
    
    f = @(x) obj(x,params);
    
    if ~aopt.mimo 
        if ~aopt.parallel
            % option 1: dfdp, not parallel, 1 output
            %----------------------------------------------------------
            [J,ip,Jo] = jaco(@obj,x0,V,0,Ord,[],{params});... df[e]   /dx [MISO]
            %objfunn = @(x) obj(x,params);[J,~] = spm_diff(objfunn,x0,1);
        else
            % option 2: dfdp, parallel, 1 output
            %----------------------------------------------------------
            [J,ip,Jo] = jacopar(@obj,x0,V,0,Ord,{params});
        end
    elseif aopt.mimo == 1
        nout   = 3;
        if ~aopt.parallel
            % option 3: dfdp, not parallel, ObjF has multiple outputs
            %----------------------------------------------------------
            [J,ip,Jo] = jaco_mimo(@obj,x0,V,0,Ord,nout,{params});
        else
            % option 4: dfdp, parallel, ObjF has multiple outputs
            %----------------------------------------------------------
            [J,ip,Jo] = jaco_mimo_par(@obj,x0,V,0,Ord,nout,{params});
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
           
    elseif aopt.mimo == 4
        % broyden method! Don't compute Jacobian but just do a rank-one
        % approximation / update
        J  = ( f(x0)*x0/(x0'*x0) ) .* aopt.J;
        J0 = f(x0)*x0;
        ip = ~~(V);
        J  = J(find(ip),:);
        Jo{:,1} = J0;
    end
    
    if aopt.mimo
        
        % Gaussian smoothing along oputput vector
        for i = 1:size(J,1)
            J(i,:) = gaufun.SearchGaussPCA(J(i,:),8);
            %J(i,:) = sign(J(i,:))'.*abs(atcm.fun.gaulinsvdfit(J(i,:)));
            %J(i,:) = atcm.fun.gausvdpca(J(i,:)',8,20);
%           [QM,GL] = AGenQn(J(i,:),8);
%           %J(i,:) = J(i,:)*QM;
%           [u,s,v] = svd(QM);
%           J(i,:) = QM*v(:,1);
        end
        %J = denan(J);
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
        
        try
            % the GaussNewton scheme needs both Grad & Hess 
            aopt.Jo      = Jo;
        end
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

function t = hypertuner(fc,t,N)
    if nargin<3; N=1000;end
    % hyperparam tuning by descent on hyperparameter
    jc = sign((fc(t) - fc(t + 1e-3)) ./ 1e-3);
    dt = t + jc*1e-3;
    nt = 0;
    
    while fc(dt) < fc(t) && nt < N
        nt = nt + 1;
        t  = dt;
        jc = sign((fc(t) - fc(t + 1e-3)) ./ 1e-3);
        dt = t + jc*1e-3;
    end
end

function [x3,J,sJ,L,D] = compute_step(df0,red,e0,step_method,params,x0,x3,df1)
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
L = 1;
D = 1;

switch search_method
        
    case 9

        a = ones(1,size(J,2));
        J = real(J);

        for i = 1:size(J,2)
            J(:,i) = rescale(J(:,i));
        end

        C = J'*J;

        a  = 1./(1+sum(C/prod(size(C))));

        %a = a(:)'.*red(:)';

        a = red(:)';

        %N = prod(size(C));
        %a = red'./N;
        x3 = a';

        dFdpp = J'*J;


       %  for i = 1:size(J,2);
       %      for j = 1:size(J,2);
       %          H(i,j) = spm_trace(J(:,i),J(:,j));
       %      end
       %  end
       % 
       %  H = H ./ norm(H);
       % 
       %  dFdpp = H;
       % 
       %  %H = (red.*H);
       % 
       % a  = 1./(1+sum(H/prod(size(H))));

        %dFdpp = spm_dx(H,J,{-4});





%         if aopt.factorise_gradients
%             a = ones(size(J,2),1);
%             [L,D] = ldl_smola(J',a);
%             dFdpp = -(L*(D./sum(diag(D)))*L');
%         else
%             dFdpp  = -(J'*J);
%         end
% 
%         [eig_vecs,D]=eig(dFdpp);
%         %eig_vals=diag(D);
%         
%         x3 = eig_vecs;
        
    
    case 8 % mirror descent with Bregman distance proximity term
        
        if aopt.factorise_gradients
            a = ones(size(J,2),1);
            [L,D] = ldl_smola(J',a);
            dFdpp = -(L*(D./sum(diag(D)))*L');
        else
            dFdpp  = -(J'*J);
        end
        
        % sometimes unstable
        dFdpp(isnan(dFdpp))=0;
        dFdpp(isinf(dFdpp))=0;
        
        % convert step size x3 (aka a) to second term in mirror descent
        proxim = (aopt.pp - x0).^2;
        if all(proxim == 0)
            proxim = 1e-3 + proxim;
            pdif = ~(aopt.pp-x0);
        else
            pdif = (aopt.pp-x0);
        end
        
        % load the whole rhs term onto x3 so that dx = x1*x3
        %x3 = J*(pdif)*(1./(2*x3')).*( proxim + 1e-3 );
        x3 = J'.*(pdif).*(1./(2*red)).*proxim;
        
    case 1
        
        if aopt.factorise_gradients
            a = ones(size(J,2),1);
            [L,D] = ldl_smola(J',a);
            dFdpp = -(L*(D./sum(diag(D)))*L');
        else
            dFdpp  = -(J'*J);
        end

        % sometimes unstable
        dFdpp(isnan(dFdpp))=0;
        dFdpp(isinf(dFdpp))=0;
        red(isnan(red))=0;
        red(isinf(red))=0;

        % Compatibility with older matlabs
        x3  = repmat(red,[1 length(red)])./(1-dFdpp);
        
        x3(isinf(x3))=0;
        x3(isnan(x3))=0;
        x3=full(x3);
        %x3 = x3 + 1e-6; % add some regularisation before (x'*x).^1/2
        
        try
            [uu,ss,vv] = spm_svd(x3);
        catch
            [uu,ss,vv] = spm_svd(denan(x3) + 1e-4);
        end
        
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
            dFdpp = -(L'*(D./sum(diag(D)))*L);
            %L = 1;
            %D = 1;
            %J = J ./ abs(sum(J));
            %dFdpp  = -(J*J');
        else
            dFdpp  = -(J*J');
        end
        
        %Initial step 
        x3  = (red)./(1-dFdpp);
        
    case {4 5}
        
        %x3 = (1/64);
        x3 = red(:);
        dFdpp = J;
        
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
        
        J = sum(dFdpp)';
        
        Cp = params.aopt.Cp;
        Cp(isnan(Cp))=0;
        Cp(isinf(Cp))=0;
        [u,s] = eig(Cp);
        s=diag(s);
        [~,order] = sort(abs(s),'descend');
        r = atcm.fun.findthenearest(cumsum(abs(s(order)))./sum(abs(s(order))),0.75);
        r = round(r);
        
        %[r = min(r,6);
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
            
            %dFdpp = ( J'./max(abs(J)') )';
            
            %dFdpp = -(dFdpp'*dFdpp);
            
            %A = trace(J'*J);
            
            %dFdpp = -(J'*J)/A;
        else
            dFdpp  = -(J'*J);
        end
                
        % Compatibility with older matlabs
        x3  = repmat(red,[1 length(red)])./(1-dFdpp);
        
        x3(isnan(x3))=0;
        x3(isinf(x3))=0;
        
        x3  = full(x3);
        x3  = x3 + 1e-3; % add some regularisation before (x'*x).^1/2
        
        % Components
        [u,s] = eig(x3);
        s     = diag(s);
        %x3    = u*diag(s.*(s < 0))*u';
        
        [~,order] = sort(abs(s),'descend');

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

sJ = dFdpp;

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
    
    if ~isvector(dx)
        % when the jacobian vector gets transposed
        dx  = x1 + (sum(a)'.*J(:));
    end
        
    %dx = x1 + (sum(a).*J')';
elseif search_method == 2
    dFdp  = a;
    dFdpp = J;
    ddx   = spm_dx(dFdpp,dFdp,{red})';    % The SPM way
    ddx   = ddx(:);
    dx    = x1 + ddx;    
elseif search_method == 3                % Rasmussen w/ varying p steps
    dx    = x1 + (a.*J);                 
    %dx = x1 + (a*J');
elseif search_method == 4                % Flat/generic descent
    dx    = x1 + (a.*J);
elseif search_method == 5
        %dx = -spm_pinv(-real(J*aopt.iS*J')-aopt.ipC*x1) * ...
        %            (-real(J*aopt.iS*aopt.er)-aopt.ipC*x1);
        
        dfdx = -spm_pinv(-real(J*aopt.iS*J')-aopt.ipC*x1);
        f    = -real(J*aopt.iS*aopt.er)-aopt.ipC*x1;
        
        [dx] = spm_dx(dfdx,f,{-2});
        dx = x1 + dx;
elseif search_method == 6 || search_method == 9
    % hyperparameter tuned step size
    dx = x1 + a.*J;
elseif search_method == 8
    dx = spm_vec(x1(:).*a(:));
    
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
X.step_method = 9;
X.im          = 1;
X.mleselect   = 0;
X.objective   = 'gauss';
X.writelog    = 0;
X.order       = 1;
X.min_df      = 0;
X.criterion   = 1e-3;
X.Q           = [];
X.inner_loop  = 1;
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
X.ismimo       = 1;
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
X.factorise_gradients = 0;
X.sample_mvn = 0;
X.do_gpr = 0;
X.normalise_gradients=0;
X.isGaussNewton = 0;
X.do_poly=0;
X.steps_choice = [];
X.isCCA = 0;
X.hypertune = 0;
X.memory_optimise = 1;
X.rungekutta = 6;
X.updateQ = 1;
X.DoMAP = 0;
X.DoMAP_Bayes = 0;
X.crit = [0 0 0 0 0 0 0 0];
X.save_constant = 0; 
X.nocheck = 0;
X.isQR = 0;
X.NatGrad = 0;
X.variance_estimation = 0;
X.gradtol = 1e-4;
X.docompare = 0;
X.isGaussNewtonReg = 1;
X.forcenewton = 0;
X.predictionerrorupdate=0;
X.simplelinesearch=0;
X.orthogradient=1;
X.rklinesearch=0;
X.lsqjacobian=0;
X.isQuasiNewton = 0;
X.verbose = 0;

% Also check if atcm is in paths ad report
try    atcm.fun.QtoGauss(1);
catch; warning(['You also need the atcm toolbox to run AO --> ' ...
    'https://github.com/alexandershaw4/atcm']);
end
    


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


%                  if n == 2
%                        k1 = Hist.dx(:,1);
%                        k2 = dx;
%                        hp = [1/8 1];
%                        fx = @(hp) (hp(1)*k1) + (hp(2)*k2);
%                        gx = @(hp) obj(fx(hp),params);
%                                               
%                        X  = fminsearch(gx,hp,options);
%                        dx = fx(X);
%                        
%                        Hist.hyperdx(1:2,n) = X;
%                        
%                  elseif n == 3
%                        k1 = Hist.dx(:,1);
%                        k2 = Hist.dx(:,2);
%                        k3 = dx;
%                        hp = [1/8 1/8 1];
%                        %dx = ((2*dt)/45) * 7*k1 + 32*k2 + 12*k3;
%                        fx = @(hp) (hp(1)*k1) + (hp(2)*k2) + (hp(3)*k3);
%                        gx = @(hp) obj(fx(hp),params);
%                        
%                        X  = fminsearch(gx,hp,options);
%                        dx = fx(X);
%                        
%                        Hist.hyperdx(1:3,n) = X;
%                        
%                  elseif n == 4
%                        k1 = Hist.dx(:,1);
%                        k2 = Hist.dx(:,2);
%                        k3 = Hist.dx(:,3);
%                        k4 = dx;
%                        hp = [1/8 1/8 1/8 1];
%                        %dx = ((2*dt)/45) * 7*k1 + 32*k2 + 12*k3 + 32*k4;
%                        fx = @(hp) (hp(1)*k1) + (hp(2)*k2) + (hp(3)*k3) + (hp(4)*k4);
%                        gx = @(hp) obj(fx(hp),params);
%                        
%                        X  = fminsearch(gx,hp,options);
%                        dx = fx(X);
%                        
%                        Hist.hyperdx(1:4,n) = X;
%                        
%                  elseif n > 4
%                        k1 = Hist.dx(:,end-3);
%                        k2 = Hist.dx(:,end-2);
%                        k3 = Hist.dx(:,end-1);
%                        k4 = Hist.dx(:,end);
%                        k5 = dx;
%                        hp = [1/8 1/8 1/8 1/8 1];
%                        %dx = ((2*dt)/45) * 7*k1 + 32*k2 + 12*k3 + 32*k4 + 7*k5;
%                        fx = @(hp) (hp(1)*k1) + (hp(2)*k2) + (hp(3)*k3) + (hp(4)*k4) + (hp(5)*k5);
%                        gx = @(hp) obj(fx(hp),params);
%                        
%                        X  = fminsearch(gx,hp,options);
%                        dx = fx(X);
%                        
%                        Hist.hyperdx(1:5,n) = X;
                 %end
                 
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