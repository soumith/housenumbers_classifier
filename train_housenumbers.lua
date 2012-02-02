require 'xlua'
xrequire ('image', true)
xrequire ('nnx', true)

----------------------------------------------------------------------
-- define network to train: CSCF
--
convnet = nn.Sequential()
convnet:add(nn.SpatialNormalization(1, image.gaussian(7)))
convnet:add(nn.SpatialConvolution(1, 8, 5, 5))
convnet:add(nn.Tanh())
convnet:add(nn.Abs())
convnet:add(nn.SpatialSubSampling(8, 4, 4, 4, 4))
convnet:add(nn.Tanh())
convnet:add(nn.SpatialConvolutionMap(nn.tables.random(8, 32, 4), 7, 7))
convnet:add(nn.Tanh())
convnet:add(nn.SpatialClassifier(nn.Linear(32,10)))

----------------------------------------------------------------------
-- training criterion: a simple Mean-Square Error
--
criterion = nn.MSECriterion()
criterion.sizeAverage = true


----------------------------------------------------------------------
-- trainer and hooks
--
optimizer = nn.SGDOptimization{module = convnet,
                               criterion = criterion,
                               learningRate = 1e-3,
                               weightDecay = 1e-6,
                               momentum = 0.8}

trainer = nn.OnlineTrainer{module = convnet, 
                           criterion = criterion,
                           optimizer = optimizer,
                           maxEpoch = 100,
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
dataHnum_train[i] = nn.DataSet{dataSetFolder='/home/rex/datasets/stanford_housenumbers/train_train_32x32/'..tostring(i-1), 
                               cacheFile='/home/rex/datasets/stanford_housenumbers/'
                               ..'train_torchcache_'..tostring(i-1),
                      channels=1,
                               useDirAsLabel = true}
      dataHnum_train[i]:shuffle()
end
dataHnum_val = {}
for i=1,10 do
dataHnum_val[i] = nn.DataSet{dataSetFolder='/home/rex/datasets/stanford_housenumbers/train_val_32x32/'..tostring(i-1), 
                               cacheFile='/home/rex/datasets/stanford_housenumbers/'
                               ..'val_torchcache_'..tostring(i-1),
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
trainData.spatialTarget = true
testData.spatialTarget = true
----------------------------------------------------------------------
-- and train !!
--
trainer:train(trainData)

