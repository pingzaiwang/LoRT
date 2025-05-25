function memo = f_D_TensorSync_Completion_v2(targetObj, sourceObj, algCfg)
    % Extract parameters and initialize variables
    sz = size(targetObj.tW);  % Tensor size
    K = sourceObj.K;          % Number of source tasks
    

    % Local estimation and aggregation at source nodes
    cW_local = cell(K, 1);
    parfor k = 1:K
        % Perform local estimation for each source task
        [cW_local{k}, ~]= f_single_task_completion_delta(sourceObj.cMask{k}, sourceObj.cY{k}, sourceObj.cW{k}, algCfg.sourceW0{k}, algCfg);
    end
    
    % Aggregate local estimates from source nodes and fused low-rank
    % minimization
    W_tgt = f_target_task_fused_completion_v2(targetObj, sourceObj, cW_local, algCfg);
    
    % Weighted averaging
    W_avg = zeros(sz);
    N = 0;
    for k = 1:K
        Nk = sum(sourceObj.cMask{k}(:)); % Number of observations in the k-th source task
        W_avg = W_avg + Nk * cW_local{k}; % Weighted sum of source tasks
        N = N + Nk;
    end
    Nt = sum(targetObj.tMask(:)); % Number of observations in the target task
    N = N + Nt;
    W_avg = W_avg + Nt * W_tgt; % Include target task in the aggregation
    W_avg = W_avg / N; % Normalize by the total number of observations
    
    % refine the target task
    algCfg.maxIter = algCfg.maxIter_D_refine;
    [W_refine, ~] = f_single_task_completion_delta(targetObj.tMask, targetObj.tY, targetObj.tW, W_avg, algCfg);
    % W_refine = W_avg + Delta;
    
    memo.tWhat = W_refine;

end

function grad = compute_gradient_DTS(tMask, tY, tW)
    % Compute the gradient of the objective function
    grad = 1/sum(tMask(:)) * tMask .* (tW - tY);
end

function [W, Delta] = f_single_task_completion_delta(tMask, tY, Wtrue, W0, algCfg)
    % Extract algorithm configuration
    lambda = algCfg.lambda0;  % Regularization parameter for nuclear norm
    gamma = algCfg.gamma;     % Learning rate
    maxIter = algCfg.maxIter; % Maximum number of iterations
    tol = algCfg.tol;         % Tolerance for convergence

    % Initialize Delta tensor
    Delta = zeros(size(W0));

    % Main iteration loop
    for iter = 1:maxIter
        % Compute the gradient
        grad = compute_gradient_DTS(tMask, tY, W0 + Delta);

        % Perform gradient descent
        Delta = Delta - gamma * grad;

        % Apply nuclear norm soft-thresholding
        Delta = f_prox_TNN(Delta, lambda * gamma);

        % Check for convergence based on changes between consecutive iterations
        if iter > 1
            maxDelta = norm(Delta(:) - Delta_last(:), 'fro');
            if maxDelta < tol
                break;
            end
        end

        % Save the Delta from the last iteration
        Delta_last = Delta;

        % Calculate RSE and PSNR for the current iteration
        RSE = norm(W0(:) + Delta(:) - Wtrue(:), 'fro') / norm(Wtrue(:), 'fro');
        PSNR = h_Psnr(Wtrue(:), W0(:) + Delta(:));

        % Output iteration information if verbose mode is enabled
        if algCfg.verbose && mod(iter, algCfg.verboseInterval) == 0
            fprintf('Iter %d: RSE = %.2e, PSNR = %.2f\n', iter, RSE, PSNR);
        end       
    end

    % Return the final estimated weight tensor
    W = W0 + Delta;
end

function [Wtgt,cWsrc] = f_target_task_fused_completion_v2(targetObj, sourceObj, cWlocal, algCfg)
    % Extract algorithm configuration
    lambda = algCfg.lambdaFuse;  % Regularization parameter for nuclear norm
    vAlpha = algCfg.vAlpha;
    gamma = algCfg.gammaFuse;    % Learning rate
    gamma1 = algCfg.gammaFuse1;
    maxIter = algCfg.maxIter;    % Maximum number of iterations
    tol = algCfg.tol;            % Tolerance for convergence

    Wtrue = targetObj.tW;        % Ground truth for target tensor
    
    K = sourceObj.K;             % Number of source tasks
    Wtgt = targetObj.tMask.*Wtrue;      % Initialize tensor W
    cDeltaSrc = cell(K,1);
    cWsrc = cell(K,1);
    for k =1:K
        cDeltaSrc{k}=Wtgt*0;
        cWsrc{k}=Wtgt*0;
    end
    % Main iteration loop
    for iter = 1:maxIter
        % Compute the gradient
        grad = zeros(size(Wtgt));
        N = 0;
        for k = 1:K
            Nk = sum(sourceObj.cMask{k}(:));
            grad = grad + Nk * (Wtgt + cDeltaSrc{k} - cWlocal{k});
            N = N + Nk;
        end
        grad = grad + targetObj.tMask .* (Wtgt - targetObj.tY);
        N = N + sum(targetObj.tMask(:));
        grad = grad / N;

        % Perform gradient descent
        oldWtgt = Wtgt;
        Wtgt = Wtgt - gamma * grad;

        % Apply nuclear norm soft-thresholding
        Wtgt = f_prox_TNN(Wtgt, lambda * gamma);
        
        for k =1:K
            grad = 1/N*sum(sourceObj.cMask{k}(:))*(cDeltaSrc{k}+oldWtgt-cWlocal{k});
            cDeltaSrc{k} = cDeltaSrc{k} - gamma1*grad;
            cDeltaSrc{k} = f_prox_TNN(cDeltaSrc{k}, vAlpha(k)*lambda * gamma1);
            cWsrc{k} = Wtgt + cDeltaSrc{k};
        end        

        % Check for convergence based on changes between consecutive iterations
        if iter > 1
            maxDelta = norm(Wtgt(:) - oldWtgt(:), 'fro');
            if maxDelta < tol
                break;
            end
        end

        % Save the W from the last iteration
        avgW = Wtgt*sum(targetObj.tMask(:));
        for k = 1:K
            avgW = avgW + sum(sourceObj.cMask{k}(:)) * cWsrc{k};
        end
        avgW = avgW / N;
        % Calculate RSE and PSNR for the current iteration
        RSE = norm(avgW(:) - Wtrue(:), 'fro') / norm(Wtrue(:), 'fro');
        PSNR = h_Psnr(Wtrue(:), avgW(:));

        % Output iteration information if verbose mode is enabled
        if algCfg.verbose && mod(iter, algCfg.verboseInterval) == 0
            fprintf('target fused: Iter %d: RSE = %.2e, PSNR = %.2f\n', iter, RSE, PSNR);
        end       
    end
end
