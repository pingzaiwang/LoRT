function [memo] = f_target_specific_refinement_completion_pgd(targetObj, algCfg)
    % f_target_specific_refinement_completion_pgd - Refines tensor completion using Projected Gradient Descent (PGD) with TNN regularization
    %
    % Input parameters:
    % targetObj - Structure containing the data for the target task, including the observed tensor, 
    %             mask, true tensor, and initial estimated tensor.
    % algCfg    - Structure containing the algorithm configuration parameters, including learning rate,
    %             regularization parameter, maximum iterations, and convergence tolerance.
    %
    % Output:
    % memo - Structure containing iteration information, including objective values, gradient norms,
    %        F-norm differences, relative squared errors (RSE), and the final estimated tensor.

    % Extract target task-related information
    tMasktgt = targetObj.tMask;  % Observation mask tensor for the target task
    tYtgt = targetObj.tY;        % Observed tensor for the target task
    tW0 = targetObj.avgW;        % Initial estimated weight tensor (from previous step)
    tWtrue = targetObj.tW;       % True weight tensor

    % Extract algorithm configuration
    lambda = algCfg.lambdaTil;   % Regularization parameter for nuclear norm
    gamma = algCfg.gamma;        % Learning rate
    maxIter = algCfg.maxIter;    % Maximum number of iterations
    tol = algCfg.tol;            % Tolerance for convergence

    % Initialize Delta (the difference between the true tensor and the estimated tensor)
    Delta = zeros(size(tW0));

    % Initialize memo structure to store results
    memo.iterCount = 0; % Actual number of iterations
    memo.objValues = zeros(maxIter, 1); % Record the objective function value for each iteration
    memo.gradNorms = zeros(maxIter, 1); % Record the gradient norm for each iteration
    memo.maxDeltaFNorm = zeros(maxIter, 1); % Maximum F-norm difference between consecutive iterations
    memo.targetFNormError = zeros(maxIter, 1); % Record the true error value for each iteration
    memo.PSNR = zeros(maxIter, 1); % Record the PSNR for each iteration
    
    fprintf("Step 2: f_target_specific_refinement_completion_pgd:\n");
    
    % Main iteration loop
    for iter = 1:maxIter
        % Compute the gradient of the objective function
        grad = 1/sum(tMasktgt(:)) * tMasktgt .* (tYtgt - tW0 - Delta);
        
        % Record the current gradient norm
        memo.gradNorms(iter) = norm(grad(:), 'fro');

        % Perform gradient descent step
        Delta = Delta + gamma * grad;

        % Apply nuclear norm soft-thresholding to promote low-rank structure
        Delta = f_prox_TNN(Delta, lambda);

        % Compute the current objective function value
        pred = f_Fnorm(tMasktgt .* (tW0 + Delta - tYtgt)) / (2 * sum(tMasktgt(:)));
        obj = pred + lambda * f_comp_TNN(Delta);
        memo.objValues(iter) = obj; % Record the objective function value

        % Compute the true error and other metrics for the current iteration
        W_estimated = tW0 + Delta; % Update the estimated tensor
        memo.targetFNormError(iter) = norm(W_estimated(:) - tWtrue(:), 'fro');
        memo.targetRSE(iter) = memo.targetFNormError(iter) / norm(tWtrue(:), 'fro');
        memo.PSNR(iter) = h_Psnr(tWtrue(:), W_estimated(:));

        % Check for convergence based on changes between consecutive iterations
        if iter > 1
            maxDelta = norm(Delta(:) - memo.DeltaLast(:), 'fro');
            memo.maxDeltaFNorm(iter) = maxDelta;
            if maxDelta < tol
                memo.iterCount = iter; % Record the actual number of iterations
                break;
            end
        end

        % Store the Delta from the current iteration for comparison in the next iteration
        memo.DeltaLast = Delta;

        % Update the actual number of iterations
        memo.iterCount = iter;

        % Output iteration information if verbose mode is enabled
        if algCfg.verbose && mod(iter, algCfg.verboseInterval) == 0
            fprintf('Iter %d: f_value = %.2e, f-norm-diff = %.2e, RSE = %.2e, PSNR = %.2f\n', ...
                iter, obj, memo.maxDeltaFNorm(iter), memo.targetRSE(iter), memo.PSNR(iter));
        end
    end

    % Store the final refined weight tensor and Delta in the memo structure
    memo.W_ts = tW0 + Delta; % Final estimated tensor
    memo.Delta_opt = Delta;  % Final Delta (difference tensor)
    memo.tWhat = tW0 + Delta; % Final refined tensor
end

function nn = f_Fnorm(X)
    % f_Fnorm - Compute the Frobenius norm squared of a tensor
    %
    % Input:
    % X - The input tensor
    %
    % Output:
    % nn - The Frobenius norm squared of the tensor

    nn = sum(X(:).^2);
end
