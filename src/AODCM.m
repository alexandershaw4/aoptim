classdef AODCM < handle
% An object wrapper for inverting Spectral Dynamic Causal Models using AO.m 
% optimisation routine.    
% 
% Example usage:
%     m = AODCM(DCM);   %<-- construct the object and autopopulate fields
%     m.optimise();     %<-- run the optimisation 
%
% Or you can flag to start the optimisation when calling the constructor:
%     m = AODCM(DCM,1);
%
% Flag if its an ERP model rather than spectral:
%     m = AODCM(DCM,-,1);
%
% If your model doesn't have logged/exponentiated params that will cause nan/infs, 
% flag it with 4th input (mine does, so i use 4th input=0)
%     m = AODCM(DCM,-,-,1);
%
% Or you can re-run the optimisation with different step_method and num
% iterations:
%     m.default_optimise([1 3 1],[10 10 10]); % run 3 optimisations: first
%     with step_method=1, then step_method=3 and then 1 again; 10
%     iterations on each of the 3 opts.
%
% AS2020

    properties
        
        DCM     % the full spec'd DCM structure
        pE      % reduced priors, based on DCM.M.pE & DCM.M.pC
        pC      % reduced prior variances from DCM.M.pC
        opts    % the options structure for the opimisation 
        X       % posterior values resulting from optim
        Ep      % same as X but in structured space
        F       % posterior objective function value (deflt: Free Energy)
        CP      % posterior parameter covariance in reduced space
        history % history from the optimisation (steps, partial derivatives)
        DD      % a helper structure for the embedded wrapdm function
        Y       % the stored data from DCM.xY.y
        V       % maps between full (parameter) space and reduced
        iserp   % switch to ERP models rather than spectral
        ftype   % swith between reduced and full model 
        hist
        params
        Pp
    end
    
    methods
        
        function obj = update_parameters(obj,P)
            % after contruction, allow updating object priors
            P        = spm_unvec( spm_vec(P), obj.DCM.M.pE);
            obj.DD.P = spm_vec(P);
            
            % also save the optimisation hisotry structure from each call
            % to the optimimser
            try obj.hist = [obj.hist; obj.history];
            catch obj.hist = obj.history;
            end
        end
        
        function obj = AODCM(DCM,do_optimise,iserp,ftype)
            % Class constructor - initates the options structure for the
            % optimisation
            obj.DCM = DCM;
            
            if nargin == 4 && ~isempty(ftype)
                % wrapped function
                obj.ftype = ftype;
            else
                % straight function
                obj.ftype = 1;
            end
            
            % Flag switch to ERP model during construction
            if nargin == 3 && iserp == 1
                obj.iserp = 1;
                fun       = @obj.wraperp;
                
                % ERP convention is to separate neural and forward parameters
                if ~isfield(obj.DCM.M.pE,'J') && isfield(obj.DCM.M,'gE')
                    fprintf('Copying gE into pE for ERP model\n');
                    obj.DCM.M.pE.J    = obj.DCM.M.gE.J;
                    obj.DCM.M.pE.L    = obj.DCM.M.gE.L;
                    obj.DCM.M.pE.Lpos = obj.DCM.M.gE.Lpos;
                    obj.DCM.M.pC.J    = obj.DCM.M.gC.J;
                    obj.DCM.M.pC.L    = obj.DCM.M.gC.L;
                    obj.DCM.M.pC.Lpos = obj.DCM.M.gC.Lpos;
                end 
                
            else
                % Use default spectral response function
                obj.iserp = 0;
                fun       = @obj.wrapdm;
            end
            
            
            DD    = obj.DCM;
            DD.SP = obj.DCM.M.pE;
            P     = spm_vec(obj.DCM.M.pE);
            V     = spm_vec(obj.DCM.M.pC);
            
            % Create mapping (cm) between full and reduced space
            cm = spm_svd(diag(V),0);
            ip = find(V);
            
            if obj.ftype == 1
                % to pass to f(ßx)
                DD.P  = P;
                DD.V  = V;
                DD.cm = cm;

                % Reduced parameter vectors -
                p = ones(length(ip),1);
                c = V(ip);

                % Essential inputs for optimisation
                opts     = AO('options');
                opts.fun = fun;
                opts.x0  = p(:);
                opts.V   = c(:);
                
            else
                DD.P  = P;
                DD.V  = V;
                DD.cm = eye(length(P));
                p     = P;
                c     = V;
                
                % Essential inputs for optimisation
                opts     = AO('options');
                opts.fun = @(p) spm_vec( feval(DCM.M.IS,spm_unvec(p,DCM.M.pE),DCM.M,DCM.xU) );
                opts.x0  = p(:);
                opts.V   = c(:); 
            end
            
            if obj.iserp; opts.y   = spm_cat(obj.DCM.xY.y);
            else;         opts.y   = spm_vec(obj.DCM.xY.y);
            end
            
            opts.inner_loop  = 10;
            opts.Q           = [];
            opts.criterion   = -inf;
            opts.min_df      = 1e-12;
            opts.order       = 2;
            opts.writelog    = 0;
            opts.objective   = 'fe';
            opts.step_method = 1;
            
            opts.BTLineSearch = 0;
            opts.hyperparams  = 1;
            %if ~isempty(obj.n_it)
            %    opts.maxit = obj.n_it;
            %end
            
            % save this read for inversion
            obj.opts = opts;
            obj.pE   = p(:);
            obj.pC   = c(:);
            obj.DD   = DD;
            obj.Y    = DCM.xY.y;
            
            
            % Begin optimisation if flagged
            if nargin == 2 && do_optimise == 1
                obj.optimise();
            end
            
        end
        
        function [yy,PP] = wrapd_gauss(obj,Px,varargin)
                [y,PP,s,t] = wrapdm(obj,Px,varargin);
                w = obj.DCM.xY.Hz;
                oy = real(y);w
                cf = fit(w.',oy,'Gauss3');

                %yy = coeffvalues(cf);
                yy = cf(w);
                
        end
        
        function [yy,PP] = wrapd_cf(obj,Px,varargin)
                [y,PP,s,t,centrefreqs] = wrapdm(obj,Px,varargin);
                
                %[~,CF] = findpeaks(real(smooth(spm_vec(y))),'NPeak',4);
                
                yy = [spm_vec(y); real(log(spm_vec(centrefreqs))./60)];
                
                %yy = [spm_vec(y); log(spm_vec(centrefreqs))./8];
            
        end
        
        function [y,PP,s,t,centrefreqs] = wrapdm(obj,Px,varargin)
            % wraps the DCM/SPM integrator function into a f(P)
            % anonymous-like function accepting a reduced parameter vector
            % and returning the model output
            %
            % Constructs the model:
            %     log( M.V*M.X.*exp(M.DD.P) ) == M.V'*M.Ep
            %
            % so that AO.m actually optimises X
            
            DD   = obj.DD;
            P    = DD.P;
            cm   = DD.cm;
            
            X0 = cm*Px(:);
            X0(X0==0)=1;
            X0 = full(X0.*exp(P(:)));
            X0 = log(X0);
            X0(isinf(X0)) = 0;
            
            PP = spm_unvec(X0,DD.SP);
            
            if isfield(PP,'J')
                % neural masses with a J parameter
                PP.J(PP.J==0)=-1000;
            end
            
            IS   = spm_funcheck(DD.M.IS);       % Integrator
            
            if nargout(IS) < 8
            % generic, works for all functions....
                y    = IS(PP,DD.M,DD.xU);
 
            elseif nargout(IS) == 8
            % this is specific to atcm.integrate3.m
                [y,w,s,g,t,pst,l,oth] = IS(PP,DD.M,DD.xU);
                s = (s{1});
                s = reshape(s,[size(s,1)*size(s,2)*size(s,3),size(s,4)]);
                jj = find(exp(PP.J));
                s = s(jj,:);
                t = pst;
               % centrefreqs = l{1}.centrals{1};
               centrefreqs=[];
            end
            
            %y    = IS(PP,DD.M,DD.xU);           % Prediction
            y    = spm_vec(y);
            y    = real(y);
            
        end
        
        function [y,PP] = wraperp(obj,Px,varargin)
            % an intermediary function to make ERP models of the form
            % { y  = g(x,P)
            % { dx = f(x,u,P,M)

            DD   = obj.DD;
            P    = DD.P;
            cm   = DD.cm;
            
            X0 = cm*Px(:);
            X0(X0==0)=1;
            X0 = full(X0.*exp(P(:)));
            X0 = log(X0);
            X0(isinf(X0)) = 0;
            
            PP = spm_unvec(X0,DD.SP);
            
            if isfield(PP,'J')
                % neural masses with a J parameter
                PP.J(PP.J==0)=-1000;
            end
            
            IS = spm_funcheck(DD.M.IS);
            
            % evaluate neural function
            [yy] = IS(PP,DD.M,DD.xU);

            % evaluate oberver func
            L = feval( DD.M.G , PP , DD.M);
            R = DD.M.R;
            
            for i = 1:length(yy)
                y{i} = R*yy{i}*L';
            end
            
            y = spm_unvec( real(spm_vec(y)), y);
            
            y0 = spm_vec(y);
            y0(isnan(y0))=0;
            y0(isinf(y0))=10000;
            
            y = spm_unvec(y0,y);
            
            %y    = spm_vec(y);
            %y    = real(y);
            
        end
        
        
        
        function [X,F,CP,Pp] = optimise(obj)
            % calls AO.m optimisation routine and returns outputs
            %
            
            [X,F,CP,Pp,History,params] = AO(obj.opts);   
            
            %close; 
            drawnow;
            
            if obj.ftype == 1
                [~, P] = obj.opts.fun(spm_vec(X));
            else
                P = X;
            end
            
            obj.X  = X;
            obj.F  = F;
            obj.CP = CP;
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
            obj.V  = obj.DD.cm;
            
            obj.history = History;
            obj.params = params;
            %obj.Pp = Pp;
            
        end
        
        function default_optimise(obj,meths,its)
            % this gets a bit meta...
            
            if nargin < 3 || isempty(its)
                its = [12 4 8];
            end
            
            if nargin < 2 || isempty(meths)
                meths = [3 1 3];
            end
            
            
            % Begin iterative optimisation loop
            for i = 1:length(meths)
                obj.opts.maxit       = its(i);
                obj.opts.step_method = meths(i);
                obj.optimise();
                
                if i ~= length(meths)
                    obj.update_parameters(obj.Ep);
                end
                
            end

            
        end
        
        function obj = nlls_optimise(obj)
            
            options = statset;
            options.Display = 'iter';
            options.TolFun  = 1e-6;
            options.MaxIter = 2;
            options.FunValCheck = 'on';
            options.DerivStep = 1e-8;

            funfun = @(b,p) full(spm_vec(spm_cat(obj.opts.fun(b.*p))));
            
            [BETA,R,J,COVB,MSE] = atcm.optim.nlinfit(obj.opts.x0,...
                            full(spm_vec(spm_cat(obj.opts.y))),funfun,full(obj.opts.x0),options);
            obj.X  = obj.V*(BETA.*obj.opts.x0);
            %obj.Ep = spm_unvec(spm_vec(obj.X),obj.DD.P);
            obj.Ep = spm_vec(obj.X);
            obj.CP = COVB;
            obj.F  = MSE;
            
        end
        
        function F = errfun(obj,f,x)
            % error / cost function - precision weighted RMSE
            
            if istable(x)
                x = x.Variables;
            end
            
            if isempty(obj.opts.Q)
               obj.opts.Q = eye(length(obj.Y));
            end
            
            Y = spm_vec(obj.opts.y);
            y = spm_vec(f(x));
            e  = (Y - y);
            Q  = {obj.opts.Q};
            nq = length(Q);
            ny = length(e);
            
%             %L(1) = real(e'*Q*e)/2; 
%             iS = sparse(0);
%             h  = sparse(length(Q),1) - log(var(spm_vec(Y))) + 4;
%             
%             for i  = 1:length(Q)
%                 iS = iS + Q{i}*(exp(-32) + exp(h(i)));
%             end
            
            er = spm_vec(Y)-spm_vec(y);
            %er = real(er'.*iS.*er)/2;
            %er(isnan(er))=inf;
            e  = ( (norm(er,2).^2)/numel(spm_vec(Y)) ).^(1/2);
            
            F    = e;%-sum(L);         
            
%             covQ = obj.opts.Q;
%             covQ(covQ<0)=0;
%             covQ = (covQ + covQ')/2;
%             
%             % pad for when using FS(y) ~= length(y)
%             padv = length(Y) - length(covQ);
%             covQ(end+1:end+padv,end+1:end+padv)=.1;
%             
%             % make sure its positive semidefinite
%             lbdmin = min(eig(covQ));
%             boost = 2;
%             covQ = covQ + ( boost * max(-lbdmin,0)*eye(size(covQ)) );
%             
%             % truth [Y] first = i.e. inclusive, mean-seeking
%             % https://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/
%             e = mvgkl(spm_vec(Y),covQ,spm_vec(y),covQ);
%             
%             e(e<0)=-e;
% %             
%             F = e;
            

        end
        
        % EXTERNAL PARAMETER ESTMIMATORS - SA, GA, SURROG, BAYES...
        %================================================================
        function mh(obj)
            % res = mh(y,x0,@genfunc,[LB,UB,params])
            %
            % Compulsory Parameters
            %   y        = data (Nx1)
            %   x0       = initial parameters (Px1)
            %   @genfunc = generative model
            %
            %   Output is nsamples*P where nsamples=njumps/sampleevery
            %
            % Example:
            %   % define a forward model (here y=a*exp(-bx))
            %   myfun=@(x,c)(exp(-x(1)*c)+x(2));
            %   % generate some noisy data
            %   true_x = [1;2];
            %   c=linspace(1,10,100);
            %   y=myfun(true_x,c) + .05*randn(1,100);
            %   % estimate parameters
            %   x0=[1;2]; % you can get x0 using nonlinear opt
            %   samples=mh(y,x0,@(x)(myfun(x,c)));
            %   figure,plot(samples)
            %
            % Other Parameters
            %   LB = lower bounds on x (default=[-inf]*ones(size(x0)))
            %   UB = upper bounds on x (default=[+inf]*ones(size(x0)))
            %   params.burnin = #burnin iterations (default=1000)
            %   params.njumps = #jumps (default=5000)
            %   params.sampleevery = keep every n jumps (default=10)
            %   params.update = update proposal every n jumps (default=20)

            fun = @(varargin)obj.wrapdm(varargin{:});
            x0  = obj.opts.x0(:);
            y   = obj.opts.y;
            
            LB = x0-2;
            UB = x0+2;
            
            pparams.burnin = 1;
            pparams.njumps = 25;
            pparams.sampleevery=10;
            pparams.update = 10;
            
            res = mh(y,x0,fun,LB,UB, pparams);
            
        end
        function vmbc(obj)
            
            % Variational Bayesian Monte Carlo
            %------------------------------------------------------------------
            fprintf('Performing Variational Bayesian Monte Carlo optimisation\n');
            LB  = (obj.opts.x0-(0.5*sqrt(obj.opts.V)));
            UB  = (obj.opts.x0+(0.5*sqrt(obj.opts.V)));
                        
            fun = @(varargin)obj.wrapdm(varargin{:});
            objective = @(x) errfun(obj,fun,x);
            
            [obj.X,obj.F,elbo_sd] = vbmc(objective,obj.opts.x0',[],[],LB',UB');
            
            [~, P] = obj.opts.fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
        end
        
        function rungekutteopt(obj,n)
            
            if nargin < 1 || isempty(n)
                n = 64;
            end
            
            % Bayesian adaptive direct search
            
            fprintf('Performing Runge-Kutta Optimisation\n');
            
            LB  = (obj.opts.x0-(0.5*sqrt(obj.opts.V)));
            UB  = (obj.opts.x0+(0.5*sqrt(obj.opts.V)));
            Px  = obj.opts.x0;
            
            dim = length(Px);
            
            fun = @(varargin)obj.wrapdm(varargin{:});
            objective = @(x) errfun(obj,fun,x);
            
            SearchAgents_no = 12;
            Max_iteration = n;
            
            %[obj.F,obj.X,cg_curve]=SCA(SearchAgents_no,Max_iteration,LB',UB',dim,objective)
            
            [obj.F,obj.X,Convergence_curve]=RUN(SearchAgents_no,Max_iteration,LB',UB',dim,objective);
            
            [~, P] = obj.opts.fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
        end
        
        function bads(obj)
            
            % Bayesian adaptive direct search
            
            fprintf('Performing Bayesian adaptive direct search\n');
            
            LB  = (obj.opts.x0-sqrt(obj.opts.V));
            UB  = (obj.opts.x0+sqrt(obj.opts.V));
            Px  = obj.opts.x0;
            
            fun = @(varargin)obj.wrapdm(varargin{:});
            objective = @(x) errfun(obj,fun,x);
            
            options.Plot = 'profile';
            
            [obj.X,obj.F] = bads(objective,Px(:)',LB',UB',LB',UB',[],options);
 
            [~, P] = obj.opts.fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
        end
        
        function kalman(obj)
            
            % Bayesian adaptive direct search
            
            fprintf('Performing Kalman Filtering Optimisation\n');
            
            Px  = obj.opts.x0;
            
            fun = @(varargin)obj.wrapdm(varargin{:});
            objective = @(x) errfun(obj,fun,x);
            
                    
            [obj.X,obj.F] = ekfopt(objective,Px(:),1e-4);

 
            [~, P] = obj.opts.fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
        end

        
        function sa(obj)
            % simulated annealing
            
            fprintf('Performing Simulated-Annealing Optimisation\n');

            LB  = (obj.opts.x0-4*sqrt(obj.opts.V));
            UB  = (obj.opts.x0+4*sqrt(obj.opts.V));
            Px  = obj.opts.x0;
            
            fun = @(varargin)obj.wrapdm(varargin{:});
            objective = @(x) errfun(obj,fun,x);

            saopts = optimoptions('simulannealbnd','MaxTime', 86400/2); % 12hours

            [obj.X,obj.F,exitFlag,output] = simulannealbnd(objective,Px,LB,UB,saopts);
            
            [~, P] = obj.opts.fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
            
        end
        
        function ga(obj)
            % genetic algorithm
            
            fprintf('Performing (Matlab) Genetic Algorithm Optimsation\n');

            LB  = (obj.opts.x0-4*sqrt(obj.opts.V));
            UB  = (obj.opts.x0+4*sqrt(obj.opts.V));
            Px  = obj.opts.x0;
            
            fun = @(varargin)obj.wrapdm(varargin{:});
            objective = @(x) errfun(obj,fun,x);


            %[X,F(i)] = ga(@optimi,length(Px),[],[],[],[],LB,UB);
            gaopts = optimoptions('ga','MaxTime', 86400/2,'PlotFcn',...
         {@gaplotbestf,@gaplotbestindiv,@gaplotexpectation,@gaplotstopping}); % 12hours
        
            [obj.X,obj.F] = ga(objective,length(Px),[],[],[],[],LB,UB,[],[],gaopts);        
            
            [~, P] = obj.opts.fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
            
        end
        
        function surrogateopt(obj)
            % surrogate optimisation algorithm
            
        % surrogate model optimisation
        %------------------------------------------------------------------
        fprintf('Performing Surrogate-Model Optimisation\n');
        
        LB  = (obj.opts.x0-(4*sqrt(obj.opts.V)));
        UB  = (obj.opts.x0+(4*sqrt(obj.opts.V)));
        Px  = obj.opts.x0;
        
        fun = @(varargin)obj.wrapdm(varargin{:});
        objective = @(x) errfun(obj,fun,x);
        
        opts1 = optimoptions('surrogateopt','PlotFcn','surrogateoptplot');
        opts1.ObjectiveLimit = 0.05;
        opts1.MaxFunctionEvaluations = 50*length(UB);
        opts1.InitialPoints=Px;
        [obj.X,obj.F] = surrogateopt(objective,LB,UB,opts1);
            
         [~, P] = obj.opts.fun(spm_vec(obj.X));
         obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);  
            
        end
        
        function ps(obj)
            % particle swarm
            
            fprintf('Performing Particle-Swarm Optimisation\n');
            
            LB  = (obj.opts.x0-4*sqrt(obj.opts.V));
            UB  = (obj.opts.x0+4*sqrt(obj.opts.V));
            Px  = obj.opts.x0;

            fun = @(varargin)obj.wrapdm(varargin{:});
            objective = @(x) errfun(obj,fun,x);
        
            options = optimoptions('particleswarm','MaxTime', 86400/2);
            options.InitialSwarmMatrix = Px';
            
            [obj.X,obj.F] = particleswarm(objective,length(Px),LB,UB,options);

            [~, P] = obj.opts.fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);  
            
        end
        
        function bayesopt(obj)
            
            % Bayesian optimsation algorithm
            %------------------------------------------------------------------
            fprintf('Performing (Matlab) Bayesian Optimsation\n');

            LB  = (obj.opts.x0-4*sqrt(obj.opts.V));
            UB  = (obj.opts.x0+4*sqrt(obj.opts.V));
            Px  = obj.opts.x0;
            
            for ip = 1:length(Px)
                name = sprintf('Par%d',ip);
                xvar(ip) = optimizableVariable(name,[LB(ip) UB(ip)],'Optimize',true);
                thename{ip} = name;
            end
            
            t = array2table(obj.opts.x0','VariableNames',thename)  ;              
            
            fun = @(varargin)obj.wrapdm(varargin{:});
            objective = @(x) errfun(obj,fun,x);

            
            reps    = 132;
            explore = 0.2;
            RESULTS = bayesopt(objective,xvar,'IsObjectiveDeterministic',true,...
                'ExplorationRatio',explore,'MaxObjectiveEvaluations',reps,...
                'AcquisitionFunctionName','expected-improvement-plus','InitialX',t);
            
            % Best Actually observed model
            % = RESULTS.MinObjective;
            obj.F   = RESULTS.MinObjective;
            obj.X   = RESULTS.XAtMinObjective.Variables;
            
            [~, P] = obj.opts.fun(spm_vec(obj.X));
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
        end
        
    end
    
end