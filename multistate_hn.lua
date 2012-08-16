require 'xlua'
xrequire ('image', true)
xrequire ('nnx', true)
require 'optim'
-- require 'cutorch'


datapath = '/misc/vlgscratch1/LecunGroup/sc3104/datasets/vision/stanford_housenumbers/jpgs/'
use_openmp = false
openmp_threads = 4
 precision = 'Double' 
-- precision =  'Float'  
extra  = 0
torch.setdefaulttensortype('torch.'..precision..'Tensor')

classes = {'0','1','2','3','4','5','6','7','8','9'}
-- geometry: width and height of input images
geometry = {32,32}
ncolor = 3 -- number of channels, 3 or 1(rgb or grayscale)
pooling = 0 -- 0 is LP, 1 is max
----------------------------------------------------------------------
-- define network to train
--
model = nn.Sequential()
-- ------------------------- DEBUG -------------------------------------
-- model:add(nn.SpatialConvolution(ncolor, 6, 5,5))
-- model:add(nn.Tanh())
-- model:add(nn.SpatialMaxPooling(2,2,2,2))
-- model:add(nn.SpatialConvolution(6, 12, 5,5))
-- model:add(nn.Tanh())
-- model:add(nn.SpatialMaxPooling(2,2,2,2))
-- model:add(nn.Reshape(12 * 5 * 5))
-- model:add(nn.Linear(12 * 5 * 5, 18))
-- model:add(nn.Tanh())
-- model:add(nn.Linear(18,#classes))

-- stage 1 : mean suppresion -> filter bank -> squashing -> pooling
model:add(nn.SpatialSubtractiveNormalization(ncolor, image.gaussian(7,7)))
-- model:add(nn.SpatialConvolution(ncolor, 16, 5, 5))
model:add(nn.SpatialConvolutionMap(nn.tables.random(3, 16, 2), 5, 5))
model:add(nn.Tanh())
if pooling == 1 then
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
else
   model:add(nn.SpatialLPPooling(16, 2, 2, 2, 2, 2))
end
model:add(nn.SpatialSubtractiveNormalization(16, image.gaussian(7,7)))
-- pipe1: conv-> tanh -> normalization
concat1 = nn.Concat(1)
pipe1 = nn.Sequential()
pipe1:add(nn.SpatialConvolutionMap(nn.tables.random(16, 512, 5), 7, 7))
pipe1:add(nn.Tanh())
if pooling == 1 then
   pipe1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
else
   pipe1:add(nn.SpatialLPPooling(512, 2, 2, 2, 2, 2))
end
pipe1:add(nn.SpatialSubtractiveNormalization(512, image.gaussian(7,7)))
pipe1:add(nn.Reshape(512*4*4))
concat1:add(pipe1)
-- pipe2: pooling
pipe2 = nn.Sequential()
if pooling == 1 then
   pipe2:add(nn.SpatialMaxPooling(2, 2, 2, 2))
else
   pipe2:add(nn.SpatialLPPooling(16, 2, 2, 2, 2, 2))
end
pipe2:add(nn.Reshape(16*7*7))
concat1:add(pipe2)
model:add(concat1)
-- -- stage 3 : standard 2-layer neural network
-- model:add(nn.Reshape(512*5*5))
model:add(nn.Linear((512*4*4) + (16*7*7), 20))
model:add(nn.Tanh())
model:add(nn.Linear(20,#classes))
model:add(nn.Tanh())

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
                               learningRate = 1e-4,
                               weightDecay = 0,
                               momentum = 0,
			       learningRateDecay = 0}

trainer = nn.OnlineTrainer{module = model, 
                           criterion = criterion,
                           optimizer = optimizer,
                           maxEpoch = 1000,
                           batchSize = 1,
                           save = 'weights_multistate',
			   timestamp = true}

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
   trainLogger:add{[' mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- run on test_set
   trainer:test(testData)

   -- print confusion
   print(confusion)
   testLogger:add{[' mean class accuracy (test set)'] = confusion.totalValid * 100}
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
color_string = ''
if ncolor == 3 then
   color_string = 'color'
end
precision_string = ''
if precision == 'Double' then 
   precision_string = 'double'
end
dataHnum_train = {}
for i=1,10 do
   dataHnum_train[i] = nn.DataSet{dataSetFolder=datapath .. 'train_train_32x32/' .. tostring(i-1), 
				  cacheFile=datapath ..'torchcache/train_torchcache_'
				     ..color_string..precision_string..tostring(i-1),
				  channels=ncolor,
				  useDirAsLabel = true}
   dataHnum_train[i]:shuffle()
end
dataHnum_val = {}
for i=1,10 do
   dataHnum_val[i] = nn.DataSet{dataSetFolder=datapath .. 'train_val_32x32/'..tostring(i-1), 
				cacheFile=datapath..'torchcache/val_torchcache_'..color_string
				   ..precision_string..tostring(i-1),
				channels=ncolor,
				useDirAsLabel = true}
   dataHnum_val[i]:shuffle()
end

if extra == 1 then
   dataHnum_extra = {}
   for i=1,10 do
      dataHnum_extra[i] = nn.DataSet{dataSetFolder=datapath .. 'extra_32x32/' .. tostring(i-1), 
				     cacheFile=datapath ..'torchcache/extra_torchcache_'
					..color_string..precision_string..tostring(i-1),
				     channels=ncolor,
				     useDirAsLabel = true}
      dataHnum_extra[i]:shuffle()
   end
   for i=1,#dataHnum_extra do
      for j=1,#dataHnum_extra[i] do
	 if (#dataHnum_extra[1][1][1])[1] == 3 then
	    dataHnum_extra[i][j][1] = image.rgb2yuv(image.rgb2nrgb(dataHnum_extra[i][j][1]:mul(0.01)))
	 else
	    dataHnum_extra[i][j][1] = dataHnum_extra[i][j][1]:mul(0.01)
	 end
      end
   end
end

for i=1,#dataHnum_train do
   for j=1,#dataHnum_train[i] do
      if (#dataHnum_train[1][1][1])[1] == 3 then
	 dataHnum_train[i][j][1] = image.rgb2yuv(image.rgb2nrgb(dataHnum_train[i][j][1]:mul(0.01)))
      else
	 dataHnum_train[i][j][1] = dataHnum_train[i][j][1]:mul(0.01)
      end
   end
end

for i=1,#dataHnum_val do
   for j=1,#dataHnum_val[i] do
      if (#dataHnum_val[1][1][1])[1] == 3 then
	 dataHnum_val[i][j][1] = image.rgb2yuv(image.rgb2nrgb(dataHnum_val[i][j][1]:mul(0.01)))
      else
	 dataHnum_val[i][j][1] = dataHnum_val[i][j][1]:mul(0.01)
      end
   end
end

trainData = nn.DataList()
testData = nn.DataList()
trainData.targetIsProbability = true
testData.targetIsProbability = true
for i=1,10 do
   trainData:appendDataSet(dataHnum_train[i],tostring(i-1))
   if extra == 1 then
      trainData:appendDataSet(dataHnum_extra[i],tostring(i-1))
   end
   testData:appendDataSet(dataHnum_val[i],tostring(i-1))
end
trainData:shuffle()

----------------------------------------------------------------------
-- and train !!
--
-- trainer:train(trainData)

