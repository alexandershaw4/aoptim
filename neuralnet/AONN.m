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
        fun        = @(m,x) imax( (x*m{1}*diag(1./(1 + exp(-m{2})))*m{3})' )-1;
        fun_nr     = @(m,x)       (x*m{1}*diag(1./(1 + exp(-m{2})))*m{3});
        g          = @(p) obj.f_nr(spm_unvec(p,m),x);
        prediction
        pred_raw  
        truth     
        op
        yy
        x
        y

    end


    methods
        
    
        function obj = AONN(y,x,nh,niter) 

            obj.x = x;
            obj.y = y;
            
            x  = full(real(x));
            [nob,np] = size(x); % num obs and vars

            ny = length(unique(y));
            obj.yy = zeros(length(y),ny);
            for i = 1:ny
                obj.yy(find(y==i-1),i)=1;
            end
            obj.truth = obj.yy;

            % weights and activations
            HL = zeros(nh,1);
            W1 = ones(np,nh)/np*nh;
            W2 = ones(nh,ny)/nh.^2;

            % accessible parameters to f
            obj.modelspace  = {W1 HL W2};

            obj.p  = real(spm_vec(obj.modelspace)) ;
            obj.c  = (~~obj.p)/32;
            obj.c  = [spm_vec(ones(size(obj.modelspace{1})))/32;
                  spm_vec(ones(size(obj.modelspace{2})))/32;
                  spm_vec(ones(size(obj.modelspace{3})))/32 ];

            % note I'm optimisming using f_nr - i.e. on a continuous, scalar
            % prediction landscape rather than binary (f)

            % free energy optimisation settings (required AO.m)
            obj.op = AO('options');
            obj.op.fun       = obj.g;
            obj.op.x0        = obj.p(:);
            obj.op.V         = obj.c(:);
            obj.op.y         = {obj.yy};
            obj.op.maxit     = niter;
            obj.op.criterion = -500;
            obj.op.step_method  = 1;
            obj.op.BTLineSearch = 0;
            obj.op.hyperparams  = 1;
            obj.op.inner_loop   = 8;

        end
        
        function obj = train(obj)
            
            [X,F,CP,Pp]    = AO(obj.op);
            obj.prediction = obj.fun(spm_unvec(X,obj.modelspace),obj.x);
            obj.pred_raw   = obj.fun_nr(spm_unvec(X,obj.modelspace),obj.x);
            obj.weightvec = X(:);
            obj.F = F;
            obj.covariance = CP;
            
        end
        
        
    end


end

