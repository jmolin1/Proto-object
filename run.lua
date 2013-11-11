#!/usr/bin/env torch
------------------------------------------------------------
-- Proto-object based saliency algorithm
--
-- Code written by: Jamal Molin
--

require 'torch'
require 'image'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'nn'
require 'inline'
require 'camera'

require 'helper_functions'
require 'image_functions'
require 'proto_functions'
require 'setup_filters'

torch.setdefaulttensortype('torch.DoubleTensor')

MAX_FILTER_SIZE = 9;
NUM_LEVELS = 1;

------------------
--GET PARAMETERS--
------------------
local defaultParameters = getDefaultParameters(NUM_LEVELS);

-------------------------------------------------------------------------
--SETUP FILTERS AND GET LOCATIONS IN NEURAL NETWORK SPATIAL CONVOLUTION--
-------------------------------------------------------------------------
local EVMSK,ODDMSK,GABORMSK,CSMSK,VMSK1,VMSK2;
conv,EVMSK,ODDMSK,GABORMSK,CSMSK,VMSK1,VMSK2 = createFilters(defaultParameters,MAX_FILTER_SIZE);

local im = image.loadJPG('/home/jamal/ProtoObject/soccer.jpg')

timerr = torch.Timer();
im = normalizeImage(im);
print('NormalizeImage = ' .. timerr:time().real);

--Generate channels from image
--timerr = torch.Timer();
local im_channels = generateChannels(im,defaultParameters,GABORMSK);
--print('GenerateChannels = ' .. timerr:time().real);

timerr = torch.Timer();
local b1Pyr_final, b2Pyr_final = makeBorderOwnership(im_channels,defaultParameters,EVMSK,ODDMSK,CSMSK,VMSK1,VMSK2);
print('MakeBorderOwnership = ' .. timerr:time().real);

timerr = torch.Timer();
local gPyr_final = makeGrouping(b1Pyr_final, b2Pyr_final, defaultParameters);
print('MakeGrouping = ' .. timerr:time().real);

--image.display{image = gPyr_final[1].subtype[1][1].data}
timerr = torch.Timer();
local h_final = ittiNorm(gPyr_final,1);
print('IttiNormalization = ' .. timerr:time().real);

--save data to file
writeToFile(h_final.data,'sal_out_map.txt');
-- make sure the data is written
image.display{image = h_final.data}
