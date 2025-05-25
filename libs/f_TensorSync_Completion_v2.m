function [memo_step1, memo_step2] = f_TensorSync_Completion_v2(targetObj, sourceObj, algCfg)
    % f_TensorSync_Completion - Complete TensorSync algorithm implementation
    %
    % Input parameters:
    % targetObj - Structure containing the data for the target task
    % sourceObj - Structure containing the data for the source tasks
    % algCfg    - Structure containing the algorithm configuration parameters
    %
    % Output parameters:
    % memo_step1 - Memo structure updated after the first step, containing results and iteration information
    % memo_step2 - Memo structure updated after the second step, containing results and iteration information

    % Extract parameters and initialize
    sz = size(targetObj.tW);  % Tensor size (dimensions of the target weight tensor)
    K = sourceObj.K;          % Number of source tasks
    gamma = algCfg.gamma;     % Learning rate for the first step
    lambda0 = algCfg.lambda0; % Regularization parameter for the nuclear norm

    % Step 1: Joint learning of target task and source tasks
    % This step performs joint low-rank completion using projected gradient descent (PGD)
    %memo_step1 = f_joint_lowrank_completion_pgd(targetObj, sourceObj, algCfg);
    memo_step1 = f_joint_lowrank_completion_pgd_v2(targetObj, sourceObj, algCfg);

    % Initialize avgW for the target task based on the joint learning results
    % avgW will be used as an initial estimate in the refinement step
    % N = sum(targetObj.tMask(:)); % Number of observed entries in the target task
    % targetObj.avgW = sum(targetObj.tMask(:)) * memo_step1.targetW; % Weighted sum of target task estimate

    % Add contributions from each source task to the initial estimate
    % for k = 1:K 
    %     targetObj.avgW = sum(sourceObj.cMask{k}(:)) * memo_step1.sourceW{k} + targetObj.avgW;
    %     N = N + sum(sourceObj.cMask{k}(:)); % Update the total number of observed entries
    % end

    % Normalize the initial estimate by the total number of observations
    targetObj.avgW = memo_step1.tWhat;

    % Step 2: Target task-specific refinement
    % This step refines the target task estimate using a specialized version of PGD
    algCfg.gamma = algCfg.gammaStep2; % Adjust learning rate for the second step
    algCfg.maxIter = algCfg.maxIter_step2;
    memo_step2 = f_target_specific_refinement_completion_pgd(targetObj, algCfg);
end
