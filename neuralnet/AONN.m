classdef AONN < handle
    
% Super simple FF NN for classification, optimised by minimising free energy
%
% [X,F,CP,f] = ao_nn(y,x,nh,niter)
%
% y  = group identifier (0,1), size(n,1)
% x  = predictor variances, size(n,m)
% nh = num neurons in hidden layer
% niter = number of iterations for optimisation
%
% AS

    properties

        weightvec  
        modelspace 
        F          
        p
        c
        covariance 
        fun_nr     = @(m,x)       (x*m{1}*diag(1./(1 + exp(-m{2})))*m{3}*(1./(1 + exp(-m{4}))) ./ sum((1./(1 + exp(-m{4})))));
        g          = @(p) obj.fun_nr(spm_unvec(p,obj.modelspace),obj.x);
        prediction
        pred_raw  
        truth     
        op
        yy
        x
        y
        yscale
    end


    methods
        
    
        function obj = AONN(y,x,nh,niter) 
            
            %obj.yscale = 1./max(abs(y(:)));
            %y = y*obj.yscale;
                        
            obj.x = x;
            obj.y = y;
            
            x  = full(real(x));
            [nob,np] = size(x); % num obs and vars

            ny = length(unique(y));
            values = unique(y);
            obj.yy = zeros(length(y),ny);
            
            obj.yy = y;
            
            if all(values == round(values))
                obj.yy = y;
                %for i = 1:ny
                    %obj.yy(find(y==i-1),i)=1;
                    %obj.yy(find(y==values(i)),i)=1;%values(i);
                %end
            else
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
                 
                
            end
            
            obj.truth = obj.yy;

            % weights and activations
            HL = zeros(nh,1);
            W1 = ones(np,nh)/np*nh;
            W2 = ones(nh,ny)/nh.^2;
            OA = ones(ny,1)/2;
            
            %HL = rand(nh,1);
            %W1 = rand(np,nh)/np*nh;
            %W2 = rand(nh,ny)/nh.^2;
            %OA = rand(ny,1);
            
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
            obj.op.y         = {obj.y};
            obj.op.maxit     = niter;
            obj.op.criterion = -500;
            obj.op.step_method  = 1;
            obj.op.BTLineSearch = 0;
            obj.op.hyperparams  = 0; % turn off 4speed for large problems
            obj.op.inner_loop   = 8;

        end
        
        function obj = train(obj)
            
            % this needs to redefined now for some reason
            obj.op.fun     = @(p) obj.fun_nr(spm_unvec(p,obj.modelspace),obj.x);
            
            obj.op.x0 = obj.p;
            obj.op.V  = obj.c;
            
            [X,F,CP,Pp]    = AO(obj.op);
            %obj.prediction = obj.fun(spm_unvec(X,obj.modelspace),obj.x);
            obj.pred_raw   = obj.fun_nr(spm_unvec(X,obj.modelspace),obj.x);
            obj.weightvec  = X(:);
            obj.F          = F;
            obj.covariance = CP;
            
            
            
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
        end
        
        function obj = updateweights(obj,w)
                 obj.p = spm_unvec(spm_vec(w),obj.p);
        end
    end
end

