clear all;clc; close all;

%initialize variables
numInput = 8; 
numHidden = 3;
numOutput = 1;
numPattern = 8;
maxEpoch = 1000;
tests = 10000;
lrate = 1;
resetCount = 0;

%generate 8 eight-dimentional random inputs and and their desired outcome
trainInputs = round(rand(numInput,numPattern));
desiredOutput = mod(sum(trainInputs,1),2)==0;

%initialize random weight matrices
w_fg = rand(numHidden,numInput)-0.5;
w_gh = rand(numOutput,numHidden)-0.5;

sse = 10000;
epoch = 1;

while sse > 0.01 
       
       %for each input, calculate output and update connection weights
       for index = 1:numPattern
            f = trainInputs(:,index);
            g = activation_fn(w_fg * f);
            h = activation_fn(w_gh * g);
            e = desiredOutput(index) - h;
         
           dw_fg = lrate*diag(g.*(1-g))*w_gh'*diag(e)*(h.*(1-h))*f';
           dw_gh = lrate*(diag(h.*(1-h))*(e'))*g';

           w_fg = w_fg + dw_fg;
           w_gh= w_gh + dw_gh;

       end
       
       %calculate SSE with the adjusted weights
       g = activation_fn(w_fg * trainInputs);
       h = activation_fn(w_gh * g);
       e = desiredOutput - h;
       sse = trace(e'*e);
       sseVec(epoch) = sse;
    
        %"brief report"
        if mod(epoch,10)==0
            epoch
            sse
        end

        %reset weights if SSE did not converge within 1000 epoch
       if maxEpoch == epoch 
         
          disp('Warning: Sum of Squared Error did NOT converge to <.01. The model will rerun with a new set of random weights')
          
          %reset sets with random weights
          w_fg = rand(numHidden,numInput)-0.5;
          w_gh = rand(numOutput,numHidden)-0.5;
          trainInputs = round(rand(numInput,numPattern));
          desiredOutput = mod(sum(trainInputs,1),2)==0;
          
          sse = 10000;
          epoch = 0;
          clearvars sseVec
          resetCount = resetCount + 1;
        end
        epoch = epoch + 1;
end

%displays how many times random weights were reset.
resetCount 


plot(sseVec);
title('SSE across epoch')
xlabel('Epoch')
ylabel('SSE')

figure
subplot(1,2,1);
imagesc(desiredOutput);  
title('Desired output')
ylabel(['Blue = Odd. ' 'Yellow = Even.'])
subplot(1,2,2);
imagesc(h);
title('Model output')

%testing the model with new patterns
for t=1:tests
      
      %new pattern
      f = round(rand(numInput,numPattern));
      desiredOutput = mod(sum(f,1),2)==0;
      
      %run model
      g = activation_fn(w_fg * f); 
      h = activation_fn(w_gh * g); 
     
      %count how many model outputs are correct
      numCorrect = sum(round(h)==desiredOutput);
      
      accuracy(t) = numCorrect/numPattern;
end
    
%display model testing results
figure
hist(accuracy);
ylabel(strcat('Frequency ', ' (',num2str(tests),' tests total)'))
title('Histogram showing distribution of accuracy rating (percentage)')
figure;
correctRates = [sum(accuracy==0) sum(accuracy==1/8) sum(accuracy==2/8) sum(accuracy==3/8) sum(accuracy==4/8) sum(accuracy==5/8) sum(accuracy==6/8) sum(accuracy==7/8) sum(accuracy==1)];
str = categorical({'0 correct'; '1 correct'; '2 correct'; '3 correct';'4 correct';'5 correct';'6 correct';'7 correct';'8 correct'});
bar(str,correctRates);  
ylabel(strcat('Frequency ', ' (',num2str(tests),' tests total)'))
title('Graph showing distribution of accuracy based on number of correct prediction')
meanAccuracy = mean(accuracy);

if meanAccuracy > 0.8
    disp(['The model successfully generalizes the new stimuli with ' num2str(meanAccuracy*100) '% accuracy']);
else
    disp(['The model generalizes the new stimuli with ' num2str(meanAccuracy*100) '% accuracy. Thus the model does not generalize well.']);
end


