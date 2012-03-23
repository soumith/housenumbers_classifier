require 'xlua'
xrequire ('image', true)
xrequire ('nnx', true)

datapath = '/misc/vlgscratch1/LecunGroup/sc3104/datasets/vision/stanford_housenumbers/'
use_openmp = true
openmp_threads = 4


torch.setdefaulttensortype('torch.FloatTensor')

classes = {'0','1','2','3','4','5','6','7','8','9'}
-- geometry: width and height of input images
geometry = {32,32}
----------------------------------------------------------------------
-- define network to train
--
model = nn.Sequential()
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolution(1, 6, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(6, 2, 2, 2, 2, 2)) -- L2
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolution(6, 16, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(16, 2, 2, 2, 2, 2)) -- L2
-- stage 3 : standard 2-layer neural network
model:add(nn.SpatialConvolution(16, 120, 5, 5))
model:add(nn.Tanh())
model:add(nn.Linear(120,#classes))

print(model)
----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
criterion = nn.DistNLLCriterion()
criterion.targetIsProbability = true

----------------------------------------------------------------------

if use_openmp then
   require 'openmp'
   openmp.setDefaultNumThreads(openmp_threads)
   print('<OpenMP> enabled with ' .. openmp_threads .. ' threads')
end



---------------------------------------------------------------------
-- trainer and hooks
--
optimizer = nn.SGDOptimization{module = model,
                               criterion = criterion,
                               learningRate = 1e-3,
                               weightDecay = 0,
                               momentum = 0.0}

trainer = nn.OnlineTrainer{module = model, 
                           criterion = criterion,
                           optimizer = optimizer,
                           maxEpoch = 20,
                           batchSize = 1,
                           save = 'weights.dat'}

confusion = nn.ConfusionMatrix{'0','1','2','3','4','5','6','7','8','9'}
trainLogger = nn.Logger('train.log')
testLogger = nn.Logger('test.log')

trainer.hookTrainSample = function(trainer, sample)
      confusion:add(trainer.module.output, sample[2])
   end

trainer.hookTestSample = function(trainer, sample)
      confusion:add(trainer.module.output, sample[2])
   end

trainer.hookTrainEpoch = function(trainer)
   -- print confusion
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- run on test_set
   trainer:test(testData)

   -- print confusion
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- plot errors
--   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
--   testLogger:style{['% mean class accuracy (test set)'] = '-'}
--   trainLogger:plot()
 --  testLogger:plot()
   end


----------------------------------------------------------------------
-- create dataset
--
dataHnum_train = {}
for i=1,10 do
dataHnum_train[i] = nn.DataSet{dataSetFolder=datapath .. 'train_train_32x32/' .. tostring(i-1), 
                               cacheFile=datapath ..'torchcache/train_torchcache_'..tostring(i-1),
                      channels=1,
                               useDirAsLabel = true}
      dataHnum_train[i]:shuffle()
end
dataHnum_val = {}
for i=1,10 do
dataHnum_val[i] = nn.DataSet{dataSetFolder=datapath .. 'train_val_32x32/'..tostring(i-1), 
                               cacheFile=datapath..'torchcache/val_torchcache_'..tostring(i-1),
                      channels=1,
                             useDirAsLabel = true}
      dataHnum_val[i]:shuffle()
end

trainData = nn.DataList()
testData = nn.DataList()
for i=1,10 do
   trainData:appendDataSet(dataHnum_train[i],tostring(i-1))
   testData:appendDataSet(dataHnum_val[i],tostring(i-1))
end
--trainData.spatialTarget = true
--testData.spatialTarget = true
----------------------------------------------------------------------
-- and train !!
--
trainer:train(trainData)

