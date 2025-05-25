function memo = f_target_only_completion_tnn_admm_v2(targetObj, algCfg)
    % f_target_only_completion_tnn_admm_v2 - Tensor completion using ADMM with TNN regularization
    %
    % Input parameters:
    % targetObj - Structure containing the data for the target task, including the true tensor,
    %             observation mask, and observed tensor.
    % algCfg    - Structure containing the algorithm configuration parameters, including rho, nu,
    %             maximum number of iterations, and tolerance for convergence.
    %
    % Output:
    % memo - Structure containing iteration information, including objective values, F-norm differences,
    %        relative squared errors (RSE), PSNR, and the final estimated tensor.

    % Extract parameters from algCfg
    rho = algCfg.rho;  % ADMM penalty parameter
    nu = algCfg.nu;    % ADMM penalty update multiplier

    % Extract tensor size and initialize variables
    sz = size(targetObj.tW);  % Size of the target tensor
    normTruth = norm(double(targetObj.tW(:)));  % Norm of the true tensor

    % Initialize tensor variables
    W = zeros(sz);  % Initial guess for the tensor
    W1 = W;         % Auxiliary variable for decoupling in ADMM

    % Initialize Lagrangian multipliers
    Y1 = W;  % Multiplier for the first constraint
    Y2 = W;  % Multiplier for the second constraint

    % Mask tensor and observation tensor
    B = targetObj.tMask;  % Observation mask
    Y = targetObj.tY;     % Observed tensor with missing entries

    % Display initial message
    fprintf('++++ f_target_only_completion_tnn_admm_v2 ++++\n');
    sz

    % Initialize memo structure to store results
    memo.iterCount = 0;  % Actual number of iterations
    memo.objValues = zeros(algCfg.maxIter, 1);  % Objective function value for each iteration
    memo.gradNorms = zeros(algCfg.maxIter, 1);  % Gradient norm for each iteration
    memo.maxWFNorm = zeros(algCfg.maxIter, 1);  % Maximum F-norm difference between consecutive iterations
    memo.targetFNormError = zeros(algCfg.maxIter, 1);  % F-norm error between estimate and true tensor
    memo.targetRSE = zeros(algCfg.maxIter, 1);  % Relative squared error (RSE) for each iteration
    memo.PSNR = zeros(algCfg.maxIter, 1);  % PSNR for each iteration

    % Main ADMM iteration loop
    for iter = 1:algCfg.maxIter
        oldL = W;  % Store the previous estimate for convergence checking

        % Update W1 (F-norm decoupler)
        W1 = (Y1 + rho * W - B .* Y2 + rho * Y) ./ (rho * B + rho);

        % Update W (TNN regularizer)
        W_tmp = W1 - Y1 / rho;
        [W, fval] = f_prox_TNN(W_tmp, 1 / rho);  % Apply the proximal operator for TNN

        % Record iteration results
        memo.iterCount = iter;
        memo.objValues(iter) = fval;
        memo.maxWFNorm(iter) = norm(double(W1(:) - oldL(:)), 'fro');
        memo.targetFNormError(iter) = norm(double(W1(:) - targetObj.tW(:)), 'fro');
        memo.targetRSE(iter) = memo.targetFNormError(iter) / normTruth;
        memo.PSNR(iter) = h_Psnr(targetObj.tW(:), W1(:));

        % Print iteration state if verbose mode is enabled
        if algCfg.verbose && mod(iter, algCfg.verboseInterval) == 0
            fprintf('Iter %d: f-norm-difference = %.2e, RSE = %.2e, PSNR = %.2f\n', ...
                iter, memo.maxWFNorm(iter), memo.targetRSE(iter), memo.PSNR(iter));
        end

        % Check for convergence
        if memo.maxWFNorm(iter) < algCfg.tol
            fprintf('Stopped: %d: f-norm-difference = %.2e, RSE = %.2e, PSNR = %.2f\n', ...
                iter, memo.maxWFNorm(iter), memo.targetRSE(iter), memo.PSNR(iter));
            memo.tWhat = W1;
            break;
        end

        % Update Lagrangian multipliers
        Y1 = Y1 + rho * (W - W1);
        Y2 = Y2 + rho * B .* (W1 - Y);

        % Update rho for the next iteration
        rho = min(rho * nu, algCfg.maxRho);
    end

    % Store the final estimate in the memo structure
    memo.tWhat = W;
end
