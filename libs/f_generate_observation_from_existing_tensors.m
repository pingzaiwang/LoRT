function [targetObj, sourceObj] = f_generate_observation_from_existing_tensors(dataCfg, tWtgt, cWsrc)
    % f_generate_observation_from_existing_tensors - Generates observations for the target and source tasks
    % based on existing tensors.
    %
    % Input:
    % dataCfg - Structure containing the configuration for data generation, including tensor size, 
    %           sampling rates, and noise levels.
    % tWtgt   - The true weight tensor for the target task.
    % cWsrc   - Cell array containing the true weight tensors for the source tasks.
    %
    % Output:
    % targetObj - Structure containing the generated data for the target task, including the true 
    %             weight tensor, observation mask, and observed tensor.
    % sourceObj - Structure containing the generated data for the source tasks, including the true 
    %             weight tensors, observation masks, and observed tensors.

    % Extract tensor size and relevant parameters from the configuration
    sz = dataCfg.tSize;           % Size of the tensor
    K  = dataCfg.nSourceTask;     % Number of source tasks
    SRTgt = dataCfg.SRTgt;        % Sampling rate for the target task
    vSRSrc = dataCfg.vSRSrc;      % Sampling rates for the source tasks
    sigNoise  = dataCfg.noiseLevelTarget;      % Noise level for the target task
    vSigNoiseSrc  = dataCfg.vNoiseLevelSrc;    % Noise levels for the source tasks

    % Set the true weight tensor for the target task
    targetObj.tW = tWtgt;  % Assign the provided tensor as the target task tensor
    fprintf("tnn(target W)=%0.2e\n", f_comp_TNN(targetObj.tW));  % Display the TNN (Tubal Nuclear Norm) of the target tensor
    
    % Generate the observation mask and observed tensor for the target task
    targetObj.SR  = SRTgt;  % Sampling rate for the target task
    targetObj.tMask = rand(sz) < SRTgt;  % Generate random observation mask based on the sampling rate
    targetObj.tY = targetObj.tMask .* (targetObj.tW + sigNoise * randn(sz));  % Observed tensor with added noise

    % Initialize cell arrays to store data for each source task
    sourceObj.cW = cell(K, 1);       % Cell array to store source task weight tensors
    sourceObj.cMask = cell(K, 1);    % Cell array to store observation masks for each source task
    sourceObj.cY = cell(K, 1);       % Cell array to store observed tensors for each source task
    sourceObj.vSR = vSRSrc;          % Vector of sampling rates for each source task
    sourceObj.K = K;                 % Number of source tasks

    % Generate data for each source task
    for k = 1:K 
        % Set the true weight tensor for the k-th source task
        sourceObj.cW{k} = cWsrc{k};  % Assign the provided tensor as the k-th source task tensor
        
        % Generate the observation mask and observed tensor for the k-th source task
        sourceObj.cMask{k} = randn(sz) < vSRSrc(k);  % Generate random observation mask based on the sampling rate
        sourceObj.cY{k} = sourceObj.cMask{k} .* (sourceObj.cW{k} + vSigNoiseSrc(k) * randn(sz));  % Observed tensor with added noise
    end 
end
