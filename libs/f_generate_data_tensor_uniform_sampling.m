function [targetObj, sourceObj] = f_generate_data_tensor_uniform_sampling(dataCfg)
    % f_generate_data_tensor_uniform_sampling - Generates synthetic tensor data with uniform sampling
    % for both target and source tasks.
    %
    % Input:
    % dataCfg - Structure containing the configuration for data generation, including tensor size, 
    %           ranks, sampling rates, and noise levels.
    %
    % Output:
    % targetObj - Structure containing the generated data for the target task, including the true 
    %             weight tensor, observation mask, and observed tensor.
    % sourceObj - Structure containing the generated data for the source tasks, including the true 
    %             weight tensors, observation masks, and observed tensors.

    % Extract tensor size and rank information from the configuration
    sz = dataCfg.tSize;            % Size of the tensor
    r  = dataCfg.tgtRank;          % Rank for the target task tensor
    rMax = dataCfg.rankMax;        % Maximum rank for the source task tensors
    K  = dataCfg.nSourceTask;      % Number of source tasks
    vH = dataCfg.vDiffTNN;         % Scaling factors to adjust the similarity between target and source tasks
    SRTgt = dataCfg.SRTgt;         % Sampling rate for the target task
    vSRSrc = dataCfg.vSRSrc;       % Sampling rates for the source tasks
    sigNoise  = dataCfg.noiseLevelTarget;     % Noise level for the target task
    vSigNoiseSrc  = dataCfg.vNoiseLevelSrc;   % Noise levels for the source tasks

    % Generate the true weight tensor for the target task with specified rank
    targetObj.tW = f_generate_low_tubal_rank_tensor(sz, r);
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
        % Generate a low-rank tensor DeltaK for the k-th source task
        DeltaK = f_generate_low_tubal_rank_tensor(sz, rMax);
        tnn = f_comp_TNN(DeltaK);  % Compute the TNN of DeltaK
        
        % Adjust the source task tensor to control similarity with the target task
        % If DeltaK's TNN is greater than the specified similarity factor, scale DeltaK accordingly
        if tnn > vH(k) 
            sourceObj.cW{k} = targetObj.tW + DeltaK * vH(k) / tnn;
        else
            sourceObj.cW{k} = targetObj.tW + DeltaK;
        end
        
        % Generate the observation mask and observed tensor for the k-th source task
        sourceObj.cMask{k} = randn(sz) < vSRSrc(k);  % Generate random observation mask based on the sampling rate
        sourceObj.cY{k} = sourceObj.cMask{k} .* (sourceObj.cW{k} + vSigNoiseSrc(k) * randn(sz));  % Observed tensor with added noise
    end 
end
