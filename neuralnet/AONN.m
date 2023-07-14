classdef AONN < handle 
% Super simple FF NN for classification, with sigmoid activations and a 
% softmax output layer, optimised by minimising free energy
%
% M = AONN(G,X,nh,niter).train
%
% G  = group identifier (0,1), size(n,1)
% x  = predictor variances, size(n,m)
% nh = num neurons in hidden layer
% niter = number of iterations for optimisation
%
% After training, assess accuracy:
% 100*(M.confusion.T.TP + M.confusion.T.TN) ./ sum(M.confusion.M(:))
%
% AS

    properties

        weightvec  
        modelspace 
        F          
        p
        c
        covariance 
        fun_nr     = @(m,x)       (x*m{1}*diag(1./(1 + exp(-m{2})))*(m{3}+m{4})*m{5}*(exp(1./(1 + exp(-m{6}))) ./ sum(exp(1./(1 + exp(-m{6}))))));
        g          = @(p) obj.fun_nr(spm_unvec(p,obj.modelspace),obj.x);
        prediction
        pred_raw  
        truth     
        op
        yy
        x
        y
        yscale
        confusion
        J
        redfun     = @(pp,x) obj.fun_nr(spm_unvec( (pp*obj.V')'.*obj.p, obj.modelspace), x)
        V
        rp
        rc
        userg
        islogistic
    end


    methods
        
    
        function obj = AONN(y,x,nh,niter,force) 
            
            %obj.yscale = 1./max(abs(y(:)));
            %y = y*obj.yscale;
            
            if nargin < 5 || isempty(force)
                force = 0;
            end
                        
            obj.x = x;
            obj.y = y;
            
            x  = full(real(x));
            [nob,np] = size(x); % num obs and vars

            if ~force
                ny = length(unique(y));
                values = unique(y);
            else
                ny = length(y);
                values = y;
            end
            
            obj.yy = zeros(length(y),ny);
            
            obj.yy = y;
            obj.islogistic=false;
            
            if all(values(:) == round(values(:))) && ~force
                obj.yy = y;
                %obj.islogistic=true;
                %for i = 1:ny
                    %obj.yy(find(y==i-1),i)=1;
                    %obj.yy(find(y==values(i)),i)=1;%values(i);
                %end
            elseif ~force
                % model the whole confusion matrix - i.e. i don't just want
                % to optimise TP and TN, but also FP and FN
                ny = size(y,1);
                obj.yy = diag(y);
                
                if numel(obj.yy) > 5000
                    % model only accuracy
                    %ny     = 1;
                    %obj.yy = y;
                    
                    [N,EDGES,BIN] = histcounts(y,round(3*log(length(y))));
                    UB = unique(BIN);
                    
                    yy = zeros(length(y),round(3*log(length(y))));
                    
                    for i = 1:length(y)
                        yy(i,BIN(i)) = 1;
                    end
                    
                    obj.yy = yy;
                    ny     = round(3*log(length(y)));
                    
                end
                
            elseif force
                
                obj.yy = y;
                
                
            end
            
            obj.truth = obj.yy;

            % weights and activations
            HL = ones(nh,1);
            W1 = ones(np,nh);
            W2 = ones(nh,ny);
            OA = ones(ny,1);
            RN = eye(nh);
            b  = eye(nh); % biases!
                        
            % accessible parameters to f - now we can refer to state space
            obj.modelspace = {W1 HL RN b W2 OA};
            obj.modelspace = spm_unvec( spm_vec(obj.modelspace)./sum(spm_vec(obj.modelspace)), obj.modelspace);

            obj.p  = real(spm_vec(obj.modelspace)) ;
            obj.c  = (~~obj.p)/32;
%             obj.c  = [spm_vec(ones(size(obj.modelspace{1})))/32;
%                  spm_vec(ones(size(obj.modelspace{2})))/32;
%                  spm_vec(ones(size(obj.modelspace{3})))/32;
%                  spm_vec(ones(size(obj.modelspace{4})))/32;
%                  spm_vec(ones(size(obj.modelspace{5})))/32;];

            % note I'm optimisming using f_nr - i.e. on a continuous, scalar
            % prediction landscape rather than binary (f)

            % free energy optimisation settings (required AO.m)
            obj.op = AO('options');
            obj.op.fun       = obj.g;
            obj.op.x0        = obj.p(:);
            obj.op.V         = obj.c(:);
            obj.op.y         = double(obj.y);
            obj.op.maxit     = niter;
            obj.op.criterion = -500;
            obj.op.step_method  = 1;
            obj.op.hyperparams  = 0; % turn off 4speed for large problems
            obj.op.inner_loop   = 8;            
            %obj.op.corrweight=1;
            obj.op.doparallel=0;
            
            obj.op.ismimo=0;
            obj.op.hypertune=0;
            obj.op.isGaussNewtonReg=0;
            obj.op.updateQ = 0;
            
            obj.op.factorise_gradients=0;
        end
        
        function obj = dpdy(obj)
            % computes numerical gradients/derivatives
            fprintf('Computing partial numerical derivatives...\n');
            y0 = obj.fun_nr(obj.modelspace,obj.x);
            p  = obj.p;
            m  = obj.modelspace;
            d  = obj.c;
            
            for i = 1:length(p)
                msp = spm_vec(m);
                msp(i) = p(i) + p(i)*d(i);
                J(i,:) = ( obj.fun_nr(spm_unvec(msp,obj.modelspace),obj.x) - y0 )./d(i);
            end
            obj.J = J;
        end
        
        function binarymodel = prune(obj)
            
            this = dpdy(obj);
            
            this = ~~this.J;
            
            binarymodel = spm_unvec(this,obj.modelspace);
            
        end
        
        function ac = accuracy(obj)
            ac = 100*(obj.confusion.T.TP + obj.confusion.T.TN) ./ sum(obj.confusion.M(:));
        end
        
        function obj = compute_covariance(obj)
            this = dpdy(obj);
            obj.covariance = (this.J*this.J')./ ((numel(this.J))-1);
        end
        
        function obj = reduce(obj,N)
            
            if isempty(obj.J)
                obj = dpdy(obj);
            end
            
            J = obj.J;
            T = clusterdata(J,'linkage','ward','maxclust',N);
            V = sparse(1:size(J,1),T,1,size(J,1),N);
            p = ones(1,N);
            v = ones(1,N)/8;
            
            obj.V = V;
            
            %obj.redfun = @(pp,x) obj.fun_nr(spm_unvec( (pp*obj.V')'.*obj.p, obj.modelspace), x)
            
            obj.rp = p;
            obj.rc = v;
            
        end
        
        function ff = logfun(obj,f,p)
            
            %f0 = @(p) f(spm_unvec(p,obj.modelspace),obj.x);
            ff = f(p);
            ff(ff<.5)=0;
            ff(ff>=.5)=1;
        end
        
        function obj = train(obj,method)
            
            if nargin < 2
                method = 1;
            end
            
            
            switch method
                case 1
                % this needs to redefined now for some reason
                obj.op.fun     = @(p) obj.fun_nr(spm_unvec(p,obj.modelspace),obj.x);
                obj.op.x0 = obj.p;
                obj.op.V  = obj.c;
                
                if obj.islogistic
                    obj.op.fun = @(p) obj.logfun(obj.op.fun,p);
                end

                case 2
                redfun    = @(pp) obj.fun_nr(spm_unvec( (pp*obj.V')'.*obj.p, obj.modelspace), obj.x);
                obj.op.fun = @(p) redfun(p);
                obj.op.x0 = obj.rp;
                obj.op.V  = obj.rc;
                
                if obj.islogistic
                    obj.op.fun = @(p) obj.logfun(obj.op.fun,p);
                end
                
            end
            
            
                        
            [X,F,CP,Pp]    = AO(obj.op);
            %obj.prediction = obj.fun(spm_unvec(X,obj.modelspace),obj.x);
            
            switch method
                case 1
                    obj.pred_raw   = obj.fun_nr(spm_unvec(X,obj.modelspace),obj.x);
                case 2
                    obj.pred_raw   = redfun(X');
            end
            
            obj.weightvec  = X(:);
            obj.F          = F;
            obj.covariance = CP;
            
            [M,T] = confustionmat([obj.truth(:) obj.pred_raw(:)]);
            obj.confusion.M = M;
            obj.confusion.T = T;
            
            ac = accuracy(obj)
        end
        
        function obj = train_bp(obj)
            
            f = @(p) obj.fun_nr(spm_unvec(p,obj.modelspace),obj.x);
            g = @(p) sum( (spm_vec(obj.y(:) - spm_vec(f(p)) )).^2);
                        
            if obj.islogistic
                  g = @(p) sum( (spm_vec(obj.y(:) - obj.logfun(f,p) )).^2);  
            end
            
            [X,F] = fminsearch(g,obj.p);
            
            obj.weightvec = X;
            obj.F = F;
            
            obj.pred_raw = obj.fun_nr(spm_unvec(obj.weightvec,obj.modelspace),obj.x);
            obj.prediction = round(...
                    obj.fun_nr(spm_unvec(obj.weightvec,obj.modelspace),obj.x));
            
            [M,T] = confustionmat([obj.truth(:) obj.prediction(:)]);
            obj.confusion.M = M;
            obj.confusion.T = T;
            
            ac = accuracy(obj)
        end
        
        function obj = updateweights(obj,w)
                 obj.p = spm_unvec(spm_vec(w),obj.p);
                 
        end
    end
end

