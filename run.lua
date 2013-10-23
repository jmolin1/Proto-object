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

------------------
--GET PARAMETERS--
------------------
local defaultParameters = getDefaultParameters(10);

-------------------------------------------------------------------------
--SETUP FILTERS AND GET LOCATIONS IN NEURAL NETWORK SPATIAL CONVOLUTION--
-------------------------------------------------------------------------
local EVMSK,ODDMSK,GABORMSK,CSMSK,VMSK1,VMSK2;
conv,EVMSK,ODDMSK,GABORMSK,CSMSK,VMSK1,VMSK2 = createFilters(defaultParameters,MAX_FILTER_SIZE);


local im = image.loadJPG('/home/jamal/ProtoObject/soccer.jpg')

im = normalizeImage(im);

--Generate channels from image
local im_channels = generateChannels(im,defaultParameters,GABORMSK);

local b1Pyr_final, b2Pyr_final = makeBorderOwnership(im_channels,defaultParameters,EVMSK,ODDMSK,CSMSK,VMSK1,VMSK2);


local gPyr_final = makeGrouping(b1Pyr_final, b2Pyr_final, defaultParameters);
--print('MakeGrouping = ' .. timerr:time().real);


local h_final = ittiNorm(gPyr_final,1);

image.display{image = h_final.data}
