classdef GAN < handle
% the generator component for turning a trained ff neural network
% (using AONN object) into a sort of gan selotaped onto the front of it
%
% obj = GAN(M,dims)
%
% wjere M is the trained NN from AONN and dims is a 3 component vector
% determining the size of the created network:
%
% dims = [7 4 10] would generate the network:
% 
%        size(m{1}) = 7,4
%        size(m{2}) = 4,1
%        size(m{3}) = 4,10
%
% such that the model is m{1} * act(diag(m{2})) * m{3}.
%
% the third component (10 in this example) / output layer should match the
% number of inputs to the trained NN.
%
% Generator:
% GG.g(GG.modelspace,spec)
%
% 
%

properties
    weightvec
    modelspace
    F
    p
    c
    covariance
    
    % the generator network
    g  ;%= @(m,x)(x*m{1}*diag(1./(1+exp(-m{2})))*m{3});
    gg ;%= @(p,x) g(spm_unvec(p,obj.modelspace),x);
    
    % connect the generator and discriminator together
    gan ;%= @(p,x) obj.M.fun_nr(obj.M.modelspace, gg(p,x) );
    
    prediction
    pred_raw
    truth
    op
    yy
    x
    y
    yscale
    M
end

methods

    function obj = GAN(M,dims)
        % instantiate the object
        
        obj.M = M;
        n = dims;
        m{1} = randn(dims(1),dims(2));
        m{2} = randn(dims(2),1);
        m{3} = randn(dims(2),dims(3));
        
        obj.modelspace = m;
        
        obj.g  = @(m,x)(x*m{1}*diag(1./(1+exp(-m{2})))*m{3});
        obj.gg = @(p,x) obj.g(spm_unvec(p,obj.modelspace),x);
        
        obj.p = spm_vec(m);
        obj.c = ~~obj.p/8;
        obj.c  = [spm_vec(ones(size(m{1})))/32;
                    spm_vec(ones(size(m{2})))/32;
                    spm_vec(ones(size(m{3})))/32 ];
        obj.weightvec = obj.p;
        obj.gan = @(p,x) obj.M.fun_nr(obj.M.modelspace, obj.gg(p,x) );
        
        obj.truth = M.truth;
        
        
        % free energy optimisation settings (required AO.m)
        obj.op = AO('options');
        obj.op.fun       = obj.gan;
        obj.op.x0        = obj.p(:);
        obj.op.V         = obj.c(:);
        obj.op.y         = {M.truth}; % get the truth from NN
        obj.op.maxit     = 18;
        obj.op.criterion = -500;
        obj.op.step_method  = 1;
        obj.op.BTLineSearch = 0;
        obj.op.hyperparams  = 0; % turn off 4speed for large problems
        obj.op.inner_loop   = 8;
    end
    
    function obj = train(obj,data)
        
        
        fg = @(p) obj.gan(p,data);
        
        obj.op.x0  = obj.p;
        obj.op.V   = obj.c;
        obj.op.fun = fg;
        
        [X,F,CP,Pp]    = AO(obj.op);
        
        %obj.prediction = obj.fun(spm_unvec(X,obj.modelspace),obj.x);
        %obj.pred_raw   = obj.fun_nr(spm_unvec(X,obj.modelspace),obj.x);
        %obj.weightvec  = X(:);
        %obj.F          = F;
        %obj.covariance = CP;
            
                
                
    end
end

end
    

