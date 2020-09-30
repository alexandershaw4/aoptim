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
% AS2020

    properties
        
        DCM     % the full spec'd DCM structure
        pE      % reduced priors, based on DCM.M.pE & DCM.M.pE
        pC      % reduced prior variances from DCM.M.pC
        opts    % the options structure for the opimisation 
        X       % posterior values resulting from optim
        Ep      % sam as X but in structured space
        F       % posterior objective function value (deflt: Free Energy)
        CP      % posterior parameter covariance in reduced space
        history % history from the optimisation (steps, partial derivatives)
        DD      % a helper structure for the embedded wrapdm function
        Y       % the stored data from DCM.xY.y
        V       % maps between full (parameter) space and reduced
    end
    
    methods
        
        function obj = update_parameters(obj,P)
            % after contruction, allow updating object priors
            P        = spm_unvec( spm_vec(P), obj.DCM.M.pE);
            obj.DD.P = spm_vec(P);
        end
        
        function obj = AODCM(DCM,do_optimise)
            % Class constructor - initates the options structure for the
            % optimisation
            obj.DCM = DCM;
            
            DD    = obj.DCM;
            DD.SP = obj.DCM.M.pE;
            P     = spm_vec(obj.DCM.M.pE);
            V     = spm_vec(obj.DCM.M.pC);
            
            % Create mapping (cm) between full and reduced space
            cm = spm_svd(diag(V));
            ip = find(V);
            
            % to pass to f(ßx)
            DD.P  = P;
            DD.V  = V;
            DD.cm = cm;
                        
            % Reduced parameter vectors -
            p = ones(length(ip),1);
            c = V(ip);
            
            
            % Essential inputs for optimisation
            opts     = AO('options');
            opts.fun = @obj.wrapdm;
            opts.x0  = p(:);
            opts.V   = c(:);
            opts.y   = spm_vec(obj.DCM.xY.y);
            
            opts.inner_loop  = 12*4;
            opts.Q           = [];
            opts.criterion   = -inf;
            opts.min_df      = 1e-12;
            opts.order       = 2;
            opts.writelog    = 0;
            opts.objective   = 'fe';
            opts.step_method = 1;

            %if ~isempty(obj.n_it)
            %    opts.maxit = obj.n_it;
            %end
            
            % save this read for inversion
            obj.opts = opts;
            obj.DCM  = DCM;
            obj.pE   = p(:);
            obj.pC   = c(:);
            obj.DD   = DD;
            obj.Y    = DCM.xY.y;
            
            % Begin optimisation if flagged
            if nargin == 2 && do_optimise == 1
                obj.optimise();
            end
            
        end
        
        
        function [y,PP] = wrapdm(obj,Px,varargin)
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
            PP.J(PP.J==0)=-1000;
            
            IS   = spm_funcheck(DD.M.IS);       % Integrator
            y    = IS(PP,DD.M,DD.xU);           % Prediction
            y    = spm_vec(y);
            y    = real(y);
            
        end
        
        function [X,F,CP,Pp] = optimise(obj)
            % calls AO.m optimisation routine and returns outputs
            %
            %
            
            [X,F,CP,Pp,History] = AO(obj.opts);   
            
            [~, P] = obj.wrapdm(spm_vec(X));
            
            obj.X  = X;
            obj.F  = F;
            obj.CP = CP;
            obj.Ep = spm_unvec(spm_vec(P),obj.DD.P);
            obj.V  = obj.DD.cm;
            
            obj.history = History;
            
        end
        
    end
    
end