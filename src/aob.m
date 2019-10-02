classdef aob < handle
% A class (object) wrapper on AO optimisation and partial (numerical) 
% differentiation functions.
%
% Example:
%
% D      = aob              % new instance of aob
% D.fun  = @(x) (1:20).^-x; % pass a function
% D.x0   = 1;               % input(s) to function
% D.xV   = 1/32;            % variance term for each input
% D.data = D.fun(2);        % data to fit (for model fitting)
%
% D.diff(1)                 % compute first order derivatives (see D.dfdx)
%
% D.optimise                % run the optimisation - then see D.X, D.F, D.Fitted
%
% AS

    properties 
        fun
        x0
        xV
        dfdx
        order
        data
        crit
        
        X
        Fit
        Fitted
    end
    
    methods
        
        % f(x)
        %------------------------------------------------------------------
        function y = run(obj)
                 y = obj.fun(obj.x0);
        end
        
        % diff
        %------------------------------------------------------------------
        function diff(obj,varargin)
                if varargin{1} && isnumeric(varargin{1})
                    obj.order = varargin{1};
                end
                
                [j,ip]   = jaco(obj.fun,obj.x0,obj.xV,0,obj.order);
                obj.dfdx = j;
            
        end
        
        % optimise
        %------------------------------------------------------------------
        function optimise(obj)
            
            if ~isempty(obj.data)
                y = obj.data;
            else
                y = 0;
            end
            
            [X,F,Cp] = AO(obj.fun,obj.x0,obj.xV,y,128,[],[],obj.crit);
            
            obj.X   = X;
            obj.Fit = F;
            obj.Fitted = obj.fun(X);
            
        end
        
    end
    
end