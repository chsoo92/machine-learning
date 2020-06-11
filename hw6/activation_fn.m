function [outputMatrix] = activation_fn(inputMatrix)
    outputMatrix = 1./(1+exp(-inputMatrix));
end