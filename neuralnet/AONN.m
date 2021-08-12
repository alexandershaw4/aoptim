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
        fun_nr     = @(m,x)       (x*m{1}*diag(1./(1 + exp(-m{2})))*m{3}*(exp(1./(1 + exp(-m{4}))) ./ sum(exp(1./(1 + exp(-m{4}))))));
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
            
            if all(values == round(values)) && ~force
                obj.yy = y;
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
            HL = zeros(nh,1);
            W1 = ones(np,nh)/np*nh;
            W2 = ones(nh,ny)/nh.^2;
            OA = ones(ny,1)/2;
            
            HL = randi([1 10],nh,1);
            W1 = randi([1 10],np,nh)/np*nh;
            W2 = randi([1 10],nh,ny)/nh.^2;
            OA = randi([1 10],ny,1);
            
            % accessible parameters to f - now we can refer to state space
            obj.modelspace  = {W1 HL W2 OA};

            obj.p  = real(spm_vec(obj.modelspace)) ;
            obj.c  = (~~obj.p)/32;
            obj.c  = [spm_vec(ones(size(obj.modelspace{1})))/32;
                  spm_vec(ones(size(obj.modelspace{2})))/32;
                  spm_vec(ones(size(obj.modelspace{3})))/32;
                  spm_vec(ones(size(obj.modelspace{4})))/32;];

            % note I'm optimisming using f_nr - i.e. on a continuous, scalar
            % prediction landscape rather than binary (f)

            % free energy optimisation settings (required AO.m)
            obj.op = AO('options');
            obj.op.fun       = obj.g;
            obj.op.x0        = obj.p(:);
            obj.op.V         = obj.c(:);
            obj.op.y         = {double(obj.y)};
            obj.op.maxit     = niter;
            obj.op.criterion = -500;
            obj.op.step_method  = 1;
            obj.op.BTLineSearch = 0;
            obj.op.hyperparams  = 0; % turn off 4speed for large problems
            obj.op.inner_loop   = 8;            
            %obj.op.corrweight=1;
            obj.op.doparallel=1;
            %obj.op.ismimo=1;
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
        
        function rnn(obj,m,x)
            
            p1 = x*m{1}*diag(1./(1 + exp(-m{2})));
            
            p2 = m{3};
            
            p3 = (exp(1./(1 + exp(-m{4}))) ./ sum(exp(1./(1 + exp(-m{4})))));
            
            y = p1*p2*p3;
            
        end        
        
        function obj = reduce(obj,N)
            
            if isempty(obj.J)
                obj = dpdy(obj);
            end
            
            J = obj.J;
%             cJ = cov(J');
%             
%             N = rank(cJ); % cov rank
%             [v,D] = eig(cJ); % decompose covariance matrix
%             DD  = diag(D); [~,ord]=sort(DD,'descend'); % sort eigenvalues
%             PCV = v(:,ord(1:N))*D(ord(1:N),ord(1:N))*v(:,ord(1:N))'; % project factorised matrix without rank deficiency
            
            %Check that the principal components are orthogonal:
            %corr((v(:,ord(1:N))'*RelCh)')
            
            %Check that the reduced cov matrix explains enough of the variance in the
            %original matrix:
            %corr(spm_vec(cov(RelCh')),spm_vec(PCV)).^2
            
            T = clusterdata(J,'linkage','ward','maxclust',N);
            V = sparse(1:size(J,1),T,1,size(J,1),N);
            p = ones(1,N);
            v = ones(1,N)/8;
            
            obj.V = V;
            
            %obj.redfun = @(pp,x) obj.fun_nr(spm_unvec( (pp*obj.V')'.*obj.p, obj.modelspace), x)
            
            obj.rp = p;
            obj.rc = v;
            
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

                case 2
                redfun    = @(pp) obj.fun_nr(spm_unvec( (pp*obj.V')'.*obj.p, obj.modelspace), obj.x);
                obj.op.fun = @(p) redfun(p);
                obj.op.x0 = obj.rp;
                obj.op.V  = obj.rc;
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
            
            [M,T] = confustionmat([obj.truth obj.pred_raw]);
            obj.confusion.M = M;
            obj.confusion.T = T;
        end
        
        function obj = train_bp(obj)
            
            f = @(p) obj.fun_nr(spm_unvec(p,obj.modelspace),obj.x);
            g = @(p) sum( (spm_vec(obj.y(:) - f(p) )).^2);
                        
            [X,F] = fminsearch(g,obj.p);
            
            obj.weightvec = X;
            obj.F = F;
            
            obj.pred_raw = obj.fun_nr(spm_unvec(obj.weightvec,obj.modelspace),obj.x);
            obj.prediction = round(...
                    obj.fun_nr(spm_unvec(obj.weightvec,obj.modelspace),obj.x));
            
            [M,T] = confustionmat([obj.truth obj.prediction]);
            obj.confusion.M = M;
            obj.confusion.T = T;
        end
        
        function obj = updateweights(obj,w)
                 obj.p = spm_unvec(spm_vec(w),obj.p);
                 
        end
    end
end

