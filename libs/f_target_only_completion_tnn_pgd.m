function [memo] = f_target_only_completion_tnn_pgd(targetObj, algCfg)
    % f_target_only_completion_tnn_pgd - Tensor completion using Projected Gradient Descent (PGD) with TNN regularization
    %
    % Input parameters:
    % targetObj - Structure containing the data for the target task, including the true tensor,
    %             observation mask, and observed tensor.
    % algCfg    - Structure containing the algorithm configuration parameters, including learning rate,
    %             regularization parameter, maximum iterations, and convergence tolerance.
    %
    % Output:
    % memo - Structure containing iteration information, including objective values, gradient norms,
    %        F-norm differences, relative squared errors (RSE), and the final estimated tensor.

    % Extract target task-related information
    tMask = targetObj.tMask;  % Sampling mask tensor for the target task
    tY = targetObj.tY;        % Observed tensor for the target task 
    tW_true = targetObj.tW;   % True weight tensor (D)
    tW_init = zeros(size(tW_true)); % Initialize weight tensor to zero

    % Extract algorithm configuration
    lambda = algCfg.lambda0;  % Regularization parameter for nuclear norm
    gamma = algCfg.gamma;     % Learning rate
    maxIter = algCfg.maxIter; % Maximum number of iterations
    tol = algCfg.tol;         % Tolerance for convergence

    % Initialize weight tensor
    tW = tW_init;  % Start with a zero tensor

    % Initialize memo structure for recording iteration metrics
    memo.iterCount = 0; % Actual number of iterations
    memo.objValues = zeros(maxIter, 1); % Record the objective function value for each iteration
    memo.gradNorms = zeros(maxIter, 1); % Record the gradient norm for each iteration
    memo.maxWFNorm = zeros(maxIter, 1); % Maximum F-norm difference between consecutive iterations
    memo.targetFNormError = zeros(maxIter, 1); % Record the true error value for each iteration
    memo.targetRSE = zeros(maxIter, 1); % Record the relative squared error (RSE) for each iteration

    fprintf(">> f_target_only_completion_tnn_pgd:\n");
    
    % Main iteration loop
    for iter = 1:maxIter
        % Compute the gradient of the objective function
        grad = zeros(size(tW_init));
        grad = grad + 1/sum(tMask(:)) * tMask .* (tY - tW);

        % Record the current gradient norm
        memo.gradNorms(iter) = norm(grad(:), 'fro');

        % Perform gradient descent step
        tW = tW + gamma * grad;

        % Apply nuclear norm soft-thresholding (promotes low-rank structure)
        tW = f_prox_TNN(tW, lambda);

        % Compute the current objective function value
        vT = tMask .* tW - tY;
        pred = sum(vT(:).^2) / (2 * sum(tMask(:)));
        obj = pred + lambda * f_comp_TNN(tW);
        memo.objValues(iter) = obj; % Record the objective function value

        % Compute the true error between the estimated and true tensors
        memo.targetFNormError(iter) = norm(tW(:) - tW_true(:), 'fro');
        memo.targetRSE(iter) = memo.targetFNormError(iter) / norm(tW_true(:), 'fro');

        % Check for convergence based on changes between consecutive iterations
        if iter > 1
            maxW = norm(tW(:) - memo.WLast(:), 'fro');
            memo.maxWFNorm(iter) = maxW;
            if maxW < tol
                memo.iterCount = iter; % Record the actual number of iterations
                break;
            end
        end

        % Update and store the weight tensor from the last iteration
        memo.WLast = tW;

        % Output iteration information (if verbose mode is enabled)
        if algCfg.verbose && mod(iter, algCfg.verboseInterval) == 0
            fprintf('Iter %d: f_value = %.2e, f-norm-difference-iteration = %.2e, RSE = %.2e\n', ...
                iter, obj, memo.maxWFNorm(iter), memo.targetRSE(iter));
        end
    end

    % Store the final refined weight tensor and output tensor in the memo structure
    memo.W_ts = tW_init + tW;  % The final estimated tensor
    memo.W_opt = tW;           % The optimized weight tensor
end
