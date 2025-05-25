function memo = f_joint_lowrank_completion_pgd_v2(targetObj, sourceObj, algCfg)
    % f_joint_lowrank_completion_pgd - Joint low-rank completion using Proximal Gradient Descent (PGD)
    %
    % Input parameters:
    % targetObj - Structure containing the data for the target task
    % sourceObj - Structure containing the data for the source tasks
    % algCfg    - Structure containing the algorithm configuration parameters
    %
    % Output:
    % memo - Structure containing iteration information, including errors and the final results

    % Extract tensor dimensions and number of tasks
    sz = size(targetObj.tW);    % Dimensions of the target task tensor
    K = sourceObj.K;            % Number of source tasks
    gamma = algCfg.gamma;       % Learning rate
    lambda0 = algCfg.lambda0;   % Regularization parameter
    vAlpha = algCfg.vAlpha;     % Alpha values for source tasks regularization
    
    % Initialize parameters
    targetW = algCfg.targetW0;  % Initial parameter tensor for the target task
    sourceDelta = algCfg.sourceW0;  % Initial parameter tensors for the source tasks (cell array)
    
    % Initialize memo structure for storing error information
    memo.iterCount = 0; % Record the actual number of iterations
    memo.targetFNormError = zeros(algCfg.maxIter, 1); % F-norm error for the target task
    memo.sourceFNormError = zeros(algCfg.maxIter, K); % F-norm error for each source task
    memo.maxDeltaFNorm = zeros(algCfg.maxIter, 1);    % Maximum F-norm difference between consecutive iterations
    memo.PSNR = zeros(algCfg.maxIter, 1); % Peak Signal-to-Noise Ratio (PSNR) for the target task

    % Proximal Gradient Descent (PGD) main loop
    for t = 1:algCfg.maxIter
        % Initialize gradient tensors
        grad_target = zeros(sz);          % Gradient tensor for the target task
        grad_sources = cell(K, 1);        % Cell array to store gradient tensors for the source tasks
        for k = 1:K
            grad_sources{k} = zeros(sz);  % Initialize each source task gradient tensor
        end
        
        % Compute gradients for source tasks
        for k = 1:K
            tMaskk = sourceObj.cMask{k};  % Observation mask for the k-th source task
            tYk = sourceObj.cY{k};        % Observed tensor for the k-th source task
            grad_sources{k} = compute_gradient_source(tMaskk, tYk, sourceDelta{k},targetW);  %targetW Compute gradient
        end
        
        % Compute gradient for the target task, including influence from source tasks
        tMask = targetObj.tMask;  % Observation mask for the target task
        tY = targetObj.tY;        % Observed tensor for the target task
        grad_target = compute_gradient(tMask, tY, targetW);  % Compute gradient for the target task
        
        % Accumulate the influence of source tasks on the target task
        for k = 1:K
            grad_target = grad_target + compute_gradient_source(sourceObj.cMask{k}, sourceObj.cY{k}, targetW,sourceDelta{k});
        end
        
        % Perform proximal gradient updates
        targetW = targetW - gamma * grad_target;  % Update target task tensor
        for k = 1:K
            sourceDelta{k} = sourceDelta{k} - gamma * grad_sources{k};  % Update each source task tensor
        end
        
        % Apply nuclear norm soft-thresholding to each tensor
        targetW = f_prox_TNN(targetW, gamma * lambda0);  % Soft-thresholding for target task tensor
        N = sum(tMask(:));  % Number of observed entries in the target task
        avgW = targetW * N;  % Start with the target task contribution
        
        % Aggregate contributions from the source tasks
        for k = 1:K
            sourceDelta{k} = f_prox_TNN(sourceDelta{k}, gamma * lambda0 * vAlpha(k));  % Soft-thresholding for source task tensor
            Nk = sum(sourceObj.cMask{k}(:));  % Number of observed entries in the k-th source task
            N = N + Nk;  % Update total number of observations
            avgW = avgW + (sourceDelta{k}+targetW) * Nk;  % Add source task contribution
        end
        avgW = avgW / N;  % Normalize the aggregated tensor
        
        % Compute F-norm errors and PSNR
        memo.targetFNormError(t) = norm(avgW(:) - targetObj.tW(:), 'fro');  % F-norm error for target task
        memo.targetRSE(t) = memo.targetFNormError(t) / norm(targetObj.tW(:), 'fro');  % Relative Squared Error (RSE)
        memo.PSNR(t) = h_Psnr(targetObj.tW(:), avgW(:));  % PSNR calculation
        
        % Compute F-norm errors for each source task
        for k = 1:K
            memo.sourceFNormError(t, k) = norm(sourceDelta{k}(:) - sourceObj.cW{k}(:), 'fro');  % F-norm error for source tasks
            memo.sourceRSE(t, k) = memo.sourceFNormError(t, k) / norm(sourceObj.cW{k}(:), 'fro');  % RSE for source tasks
        end
        
        % Compute the maximum F-norm difference between consecutive iterations
        if t > 1
            delta_target = norm(targetW(:) - memo.targetWLast(:), 'fro');  % Change in target task tensor
            delta_sources = zeros(K, 1);  % Array to store changes in source task tensors
            for k = 1:K
                delta_sources(k) = norm(sourceDelta{k}(:) - memo.sourceWLast{k}(:), 'fro');  % Change in source task tensor
            end
            memo.maxDeltaFNorm(t) = max([delta_target; delta_sources]);  % Maximum change across all tensors
            
            % Check for convergence
            if memo.maxDeltaFNorm(t) < algCfg.tol
                memo.iterCount = t;  % Record the actual number of iterations
                break;
            end
        end
        
        % Store the results of the last iteration for comparison in the next iteration
        memo.targetWLast = targetW;
        memo.sourceWLast = sourceDelta;
        
        % If verbose mode is enabled, output error information every few iterations
        if algCfg.verbose && mod(t, algCfg.verboseInterval) == 0
            fprintf('Iter %d: Target RSE = %0.2e, PSNR = %0.2f\n', t, memo.targetRSE(t), memo.PSNR(t));
            for k = 1:K
                fprintf('Source Task %d RSE = %0.2e\n', k, memo.sourceRSE(t, k));
            end
            fprintf('F-norm-diff-iter = %0.2e\n', memo.maxDeltaFNorm(t));
        end
        
        % Update the actual number of iterations
        memo.iterCount = t;
    end
    
    % Store the final results in the memo structure
    memo.targetW = targetW;
    memo.sourceW = sourceDelta;
    memo.tWhat = avgW;  % The final estimated tensor after aggregation
end

function grad = compute_gradient_source(tMask, tY, tW,tWtgt)
    % compute_gradient - Compute the gradient of tensor W
    % 
    % Input parameters:
    % tMask - Observation mask tensor
    % tY - Observed tensor
    % tW - Current weight tensor
    %
    % Output:
    % grad - Computed gradient tensor
    
    grad = 1/sum(tMask(:)) * (tMask .* (tW+tWtgt - tY));
end
function grad = compute_gradient(tMask, tY, tW)
    % compute_gradient - Compute the gradient of tensor W
    % 
    % Input parameters:
    % tMask - Observation mask tensor
    % tY - Observed tensor
    % tW - Current weight tensor
    %
    % Output:
    % grad - Computed gradient tensor
    
    grad = 1/sum(tMask(:)) * (tMask .* (tW - tY));
end
