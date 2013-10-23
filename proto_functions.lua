----------------------------------------
--PROTO-OBJECT MODEL-RELATED FUNCTIONS--
----------------------------------------
--
--function: Get Default Parameters
--
function getDefaultParameters(mxLevel)

   local local_minLevel = 1;
   local local_maxLevel = mxLevel;
   local local_downsample = 'half';
   local ori = torch.DoubleTensor({0,45});
   
   local local_oris = deg2rad(torch.DoubleTensor({ori[1], ori[2], ori[1] + 90, ori[2] + 90}));
   local local_lambda = 4;
   local local_odd_lambda = 3;
   local local_even_lambda = 3;
   
   local local_gabor_lambda = 8;
   
   local local_sigma1,local_sigma2 = calcSigma(1,2);
   local msk = makeCenterSurround(local_sigma1,local_sigma2);
   local temp = msk[{ {math.ceil(msk:size()[1] / 2)},{} }];
   temp[torch.gt(temp,0)] = 1;
   temp[torch.lt(temp,0)] = -1;
   local temp_length = torch.numel(temp);
   local zc = temp[{{}, {math.ceil(msk:size()[2] / 2), temp_length - 1}, {} }] - temp[{{}, {math.ceil(msk:size()[1] / 2) + 1, temp_length}}];
   local temp_R0 = torch.eq(torch.abs(zc),2);
   
   local val = torch.numel(temp_R0);
   local local_R0;
   
   local idx = 1;
   while (idx <= val) do
      local temp_val = (temp_R0[1][idx]);
      if( temp_val == 1) then
         local_R0 = idx;
      end
      idx = idx + 1;
   end
      
   local params =  {
      channels = 'ICO',
      maxLevel = local_maxLevel,
      evenCellsPrs = {minLevel = local_minLevel,
                      maxLevel = local_maxLevel,
                      oris = local_oris,
                      numOri = local_oris:size(1),
                      lambda = local_even_lambda,
                      sigma = 0.56 * local_even_lambda,
                      gamma = 0.5
                   },

      oddCellsPrs = {minLevel = local_minLevel,
                      maxLevel = local_maxLevel,
                      oris = local_oris,
                      numOri = local_oris:size(1),
                      lambda = local_odd_lambda,
                      sigma = 0.56 * local_odd_lambda,
                      gamma = 0.5
                  },
      
      gaborPrs = {lambda = local_gabor_lambda,
                  sigma = 0.4 * local_gabor_lambda,
                  gamma = 0.8,
                  oris = torch.DoubleTensor({0,math.pi/4,math.pi/2,3*math.pi/4}),
                  numOri = 4
                 },

      csPrs = { downSample = local_downsample,
                inner = local_sigma1,
                outer = local_sigma2,
                depth = local_maxLevel
             },

      bPrs = { minLevel = local_minLevel,
               maxLevel = local_maxLevel,
               numOri = local_oris:size(1),
               alpha = 1,
               oris = local_oris,
               CSw = 1
            },

      vmPrs = { minLevel = local_minLevel,
                maxLevel = local_maxLevel,
                oris = local_oris,
                numOri = local_oris:size(1),                
                R0 = local_R0
             },
      
      giPrs = { w_sameChannel = 1
            }
   }
   
   return params;
end

--
--function: Make Colors
--
function makeColors(im)

   local r = torch.DoubleTensor(im[1]);
   local g = torch.DoubleTensor(im[2]);
   local b = torch.DoubleTensor(im[3]);
   
   local gray = torch.div(torch.add(torch.add(r, g),b), 3);

   local msk = torch.DoubleTensor(gray):clone();
   msk[torch.lt(msk,torch.max(gray) * 0.1)] = 0;
   msk[torch.ne(msk,0)]=1;
  
   r = safeDivide(torch.cmul(r,msk),gray:clone());
   g = safeDivide(torch.cmul(g,msk),gray:clone());
   b = safeDivide(torch.cmul(b,msk),gray:clone());

   local R = r - torch.div(g + b,2);
   R[torch.lt(R,0)] = 0;

   local G = g - torch.div(r + b,2);
   G[torch.lt(G,0)] = 0;
   
   local B = b - torch.div(r + g,2);
   B[torch.lt(B,0)] = 0;

   local Y = torch.div(r + g,2) - torch.div(torch.abs(r - g),2) - b;   
   Y[torch.lt(Y,0)] = 0;
   
   return gray,R,G,B,Y;

end

--
--function: Generate Channels
--
function generateChannels(img, params, fil_vals)
   --Get different feature channels
   local gray,R,G,B,Y =  makeColors(img);
   
   --Generate color opponency channels
   local rg = R - G;
   local by = B - Y;
   local gr = G - R;
   local yb = Y - B;
      
   --Threshold opponency channels
   rg[torch.lt(rg,0)] = 0;
   gr[torch.lt(gr,0)] = 0;
   by[torch.lt(by,0)] = 0;
   yb[torch.lt(yb,0)] = 0;

   local return_result = {};

   --Set channels
   for c = 1,#params.channels do
      --Intensity Channel
      if params.channels:sub(c,c) == 'I' then
         return_result[c] = {type = 'Intensity', subtype = { {data = torch.DoubleTensor(gray), type =  'Intensity'} } }

      --Color Channel
      elseif params.channels:sub(c,c) == 'C' then
            return_result[c] = {type = 'Color', subtype = { {data = torch.DoubleTensor(rg), type = 'Red-Green Opponency'}, 
                                                            {data = torch.DoubleTensor(gr), type = 'Green-Red Opponency'},
                                                            {data = torch.DoubleTensor(by), type = 'Blue-Yellow Opponency'},
                                                            {data = torch.DoubleTensor(yb), type = 'Yellow-Blue Opponency'} } }
      --Orientation Channel
      elseif params.channels:sub(c,c) == 'O' then
         
            return_result[c] = {type = 'Orientation', subtype = { {data = torch.DoubleTensor(gray), ori = 0, type = 'Orientation', filval = fil_vals[1][1]}, 
                                                                  {data = torch.DoubleTensor(gray), ori = math.pi/4, type = 'Orientation', filval = fil_vals[1][2]},
                                                                  {data = torch.DoubleTensor(gray), ori = math.pi/2, type = 'Orientation', filval = fil_vals[1][3]},
                                                                  {data = torch.DoubleTensor(gray), ori = 3 * math.pi/4, type = 'Orientation', filval = fil_vals[1][4]} } }
      end
   end
   
   return return_result;
   
end

--
--function: Make Pyramid
--
function makePyramid(img, params)
   local depth = params.maxLevel;
   local pyr = {};
   local curr_width = img:size(2);
   local curr_height = img:size(1);
   
   pyr[1] = {data = img:clone()};
   for level = 2,depth do
      if params.csPrs.downSample == 'half' then
         curr_width = math.ceil(curr_width * 0.7071)
         curr_height = math.ceil(curr_height * 0.7071)
         pyr[level] = {data = image.scale(pyr[level-1].data,curr_width,curr_height,'simple')};

      elseif params.csPrs.downSample == 'full' then
         curr_width = math.ceil(curr_width * 0.5)
         curr_height = math.ceil(curr_height * 0.5)
         pyr[level] = {data = image.scale(pyr[level-1].data,curr_width,curr_height,'simple')};
      else
         print('Please specify if downsampling should be half or full octave');
      end
      
   end
   return pyr;
end


--
--function: Make Even Orientation Cells
--
function makeEvenOrientationCells(theta,lambda,sigma,gamma)
   --Decode inputs
   local sigma_x = sigma;
   local sigma_y = sigma/gamma;
   
   --Bounding box
   local nstds = 1;
   local xmax = math.max(math.abs(nstds*sigma_x*math.cos(math.pi/2-theta)),math.abs(nstds*sigma_y*math.sin(math.pi/2-theta)));
   xmax = math.ceil(math.max(1,xmax));
   local ymax = math.max(math.abs(nstds*sigma_x*math.sin(math.pi/2-theta)),math.abs(nstds*sigma_y*math.cos(math.pi/2-theta)));
   ymax = math.ceil(math.max(1,ymax));
   
   --Meshgrid
   local xmin = -xmax;
   local ymin = -ymax;
   local xsize = xmax*2 + 1;
   local ysize = ymax*2 + 1;
   local x_array = torch.DoubleTensor(1,xsize);
   local i = xmin-1; x_array:apply(function() i=i+1;return i end);
   local y_array = torch.DoubleTensor(ysize,1);
   local i = ymin-1; y_array:apply(function() i=i+1;return i end);
   local x,y;
   x,y = meshGrid(x_array,y_array);

   --Rotation
   local x_theta = torch.add(torch.mul(x,math.cos(math.pi/2-theta)), torch.mul(y,math.sin(math.pi/2-theta)));
   local y_theta = torch.add(torch.mul(x,-1*math.sin(math.pi/2-theta)), torch.mul(y,math.cos(math.pi/2-theta)));

   local msk = torch.exp(torch.add(torch.div(torch.pow(x_theta,2),sigma_x^2),torch.div(torch.pow(y_theta,2),sigma_y^2)):mul(-0.5)):mul(1/(2*math.pi*sigma_x*sigma_y)):cmul(torch.cos(torch.mul(x_theta,2*math.pi/lambda)));
   msk = msk - torch.mean(msk);
   
   return msk;
end

--
--function: Edge Even Pyramid
--
function edgeEvenPyramid(map,params,fil_vals)
   local prs = params.evenCellsPrs;   
   
   --Initialize newMap
   local newMap = {};
   for i = prs.minLevel,prs.maxLevel do
      newMap[i] = { orientation = {} };
   end
   --timerr = torch.Timer();
   local temp_data,outputs;
   for level = prs.minLevel,prs.maxLevel do
      temp_data = torch.DoubleTensor(map[level].data);
      temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
      
      conv:forward(temp_data)
      outputs = conv.output;

      for local_ori = 1,prs.numOri do
         newMap[level].orientation[local_ori] = {ori = prs.oris[local_ori], data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_vals[1][local_ori]])};
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
      end
   end
   --print('Time elapse = ' .. timerr:time().real);

   return newMap;
end

--
--function: Make Odd Orientation Cells
--
function makeOddOrientationCells(theta,lambda,sigma,gamma)
   --Decode inputs
   local sigma_x = sigma;
   local sigma_y = sigma/gamma;
   
   --Bounding box
   local nstds = 1;
   local xmax = math.max(math.abs(nstds*sigma_x*math.cos(math.pi/2-theta)),math.abs(nstds*sigma_y*math.sin(math.pi/2-theta)));
   xmax = math.ceil(math.max(1,xmax));
   local ymax = math.max(math.abs(nstds*sigma_x*math.sin(math.pi/2-theta)),math.abs(nstds*sigma_y*math.cos(math.pi/2-theta)));
   ymax = math.ceil(math.max(1,ymax));
   
   --Meshgrid
   local xmin = -xmax;
   local ymin = -ymax;
   local xsize = xmax*2 + 1;
   local ysize = ymax*2 + 1;
   local x_array = torch.DoubleTensor(1,xsize);
   local i = xmin-1; x_array:apply(function() i=i+1;return i end);
   local y_array = torch.DoubleTensor(ysize,1);
   local i = ymin-1; y_array:apply(function() i=i+1;return i end);
   local x,y;
   x,y = meshGrid(x_array,y_array);
   
   --Rotation
   local x_theta = torch.add(torch.mul(x,math.cos(math.pi/2-theta)), torch.mul(y,math.sin(math.pi/2-theta)));
   local y_theta = torch.add(torch.mul(x,-1*math.sin(math.pi/2-theta)), torch.mul(y,math.cos(math.pi/2-theta)));

   local msk1 = torch.exp(torch.add(torch.div(torch.pow(x_theta,2),sigma_x^2),torch.div(torch.pow(y_theta,2),sigma_y^2)):mul(-0.5)):mul(1/(2*math.pi*sigma_x*sigma_y)):cmul(torch.sin(torch.mul(x_theta,2*math.pi/lambda)));
   msk1 = torch.DoubleTensor(msk1 - torch.mean(msk1)):clone();
   
   local msk2 = torch.exp(torch.add(torch.div(torch.pow(x_theta,2),sigma_x^2),torch.div(torch.pow(y_theta,2),sigma_y^2)):mul(-0.5)):mul(1/(2*math.pi*sigma_x*sigma_y)):cmul(torch.sin(torch.mul(x_theta,2*math.pi/lambda) + math.pi));
   msk2 = torch.DoubleTensor(msk2 - torch.mean(msk2)):clone();
   
   return msk1, msk2;
end

--
--function: Edge Odd Pyramid
--
function edgeOddPyramid(map,params,fil_vals)
   local prs = params.oddCellsPrs;
   
   --Initialize newMap
   local newMap1 = {}
   local newMap2 = {}
   
   for i = prs.minLevel,prs.maxLevel do
      newMap1[i] = { orientation = {} }
      newMap2[i] = { orientation = {} }
   end
   
   local temp_data, outputs;
   for level = prs.minLevel,prs.maxLevel do
      temp_data = torch.DoubleTensor(map[level].data);
      temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
      
      conv:forward(temp_data)
      outputs = conv.output;
      for local_ori = 1,prs.numOri do
         newMap1[level].orientation[local_ori] = {ori = prs.oris[local_ori], data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_vals[1][local_ori]])};
         --newMap2[level].orientation[local_ori] = {ori = prs.oris[local_ori], data = {1,2;3,4} };
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
      end
   end
   
   return newMap1,newMap2;
end

--
--function: Make Complex Edge
--
function makeComplexEdge(EPyr, OPyr)
   local cPyr = {};

   for i = 1,#EPyr do
      cPyr[i] = { orientation = {} }
   end

   for level = 1,#EPyr do
      for local_ori = 1,#EPyr[level].orientation do
         cPyr[level].orientation[local_ori] = {data = torch.sqrt(torch.pow(torch.Tensor(EPyr[level].orientation[local_ori].data),2) + torch.pow(torch.Tensor(OPyr[level].orientation[local_ori].data),2)) };
      end
   end
   
   return cPyr;
end

--
--function: Gabor Pyramid
--
function gaborPyramid(pyr,ori,params, fil_val)
   local depth = params.maxLevel;
   local gaborPrs = params.gaborPrs;
   local Evmsk = makeEvenOrientationCells(ori,gaborPrs.lambda,gaborPrs.sigma,gaborPrs.gamma);
   local gaborPyr = {};
   for i = 1,depth do
      gaborPyr[i] = { data = {} }
   end
 
   local temp_data;
   for level = 1,depth do
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      temp_data = torch.DoubleTensor(pyr[level].data);
      temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
      conv:forward(temp_data);
      outputs = conv.output;
      gaborPyr[level].data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_val]);
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
   end
   return gaborPyr;
end

--
--function Make Gauss
--
function makeGauss(dim1,dim2,sigma_1,sigma_2,theta)
   local x0 = 0;
   local y0 = 0;
   local norm = 1;
   local msk = torch.DoubleTensor(dim1:size()[1],dim2:size()[1]):zero();
   
   local xmax = dim1[#dim1];
   local ymax = dim2[#dim2];
   
   local xmin = -xmax;
   local ymin = -ymax;
   local xsize = xmax*2 + 1;
   local ysize = ymax*2 + 1;
   local x_array = torch.DoubleTensor(1,xsize);
   local i = xmin-1; x_array:apply(function() i=i+1;return i end);
   local y_array = torch.DoubleTensor(ysize,1);
   local i = ymin-1; y_array:apply(function() i=i+1;return i end);
   
   local X,Y;
   X,Y = meshGrid(x_array,y_array);

   local a = math.cos(theta)^2/2/sigma_1^2 + math.sin(theta)^2/2/sigma_2^2;
   local b = -math.sin(theta)^2/4/sigma_1^2 + math.sin(2*theta)^2/4/sigma_2^2;
   local c = math.sin(theta)^2/2/sigma_1^2 + math.cos(theta)^2/2/sigma_2^2;
   
   if norm then
      msk = torch.mul(torch.exp(torch.mul(torch.mul(torch.pow(X-x0,2),a) + torch.mul(torch.cmul(X-x0,Y-y0),2*b) + torch.mul(torch.pow(Y-y0,2),c),-1)),1/(2*math.pi*sigma_1*sigma_2)) ;
   else
      msk = torch.exp(torch.mul(torch.mul(torch.pow(X-x0,2),a) + torch.mul(torch.cmul(X-x0,Y-y0),2*b) + torch.mul(torch.pow(Y-y0,2),c),-1));
   end
   
   return msk;
end

--
--function: Make Center Surround
--
function makeCenterSurround(std_center, std_surround)
   --get dimensions
   local center_dim = math.ceil(3*std_center);
   local surround_dim = math.ceil(3*std_surround);
   --create gaussians
   local idx_center = torch.range(-center_dim,center_dim,1);
   local idx_surround = torch.range(-surround_dim,surround_dim,1);
   local msk_center = makeGauss(idx_center,idx_center,std_center,std_center,0);
   local msk_surround = makeGauss(idx_surround,idx_surround,std_surround,std_surround,0);
   --difference of gaussians
   local msk = torch.mul(msk_surround,-1);
   local temp = torch.add(msk[{{surround_dim+1-center_dim,surround_dim+1+center_dim},{surround_dim+1-center_dim,surround_dim+1+center_dim}}],msk_center);
   msk[{{surround_dim+1-center_dim,surround_dim+1+center_dim},{surround_dim+1-center_dim,surround_dim+1+center_dim}}] = temp;
   msk = msk - (torch.sum(msk) / (msk:size()[1] * msk:size()[2]))

   return msk;
end

--
--function: CS Pyramid
--
function csPyramid(pyr,params,fil_val)
   local depth = params.maxLevel;
   local csPrs = params.csPrs;
   local CSmsk = makeCenterSurround(csPrs.inner,csPrs.outer);
   local csPyr = {};
   for i = 1,depth do
      csPyr[i] = { data = {} }
   end

   local temp_data;
   for level = 1,depth do
      ------------------------------------------------------------------------------   
      ------------------------------------------------------------------------------
      temp_data = torch.DoubleTensor(pyr[level].data);
      temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
      conv:forward(temp_data);
      outputs = conv.output;
      csPyr[level].data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_val]);
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
   end
   
   return csPyr;
end

--
--funciton: Separate Pyramids
--
function separatePyr(pyr)
   local pyr1 = {};
   local pyr2 = {};
   for i = 1,#pyr do
      pyr1[i] = { data = {} }
      pyr2[i] = { data = {} }
   end
   for level = 1,#pyr do
      pyr1[level].data = torch.DoubleTensor(pyr[level].data);
      pyr2[level].data = torch.DoubleTensor(pyr[level].data) * -1;
      pyr1[level].data[torch.lt(pyr1[level].data,0)] = 0;
      pyr2[level].data[torch.lt(pyr2[level].data,0)] = 0;
   end

   return pyr1,pyr2;
end

--
--function: Sum pyramids
--
function sumPyr(pyr1,pyr2)
   local pyr = {};
   
   for i = 1,#pyr1 do
      if (pyr1[1].orientation ~= nil) then
         pyr[i] = {orientation = {}};
         for o = 1,#pyr1[i].orientation do
            if (pyr1[i].orientation[o].ori ~= nil) then               
               if (pyr1[i].orientation[1].invmsk ~= nil) then
                  pyr[i].orientation[o] = {ori = {}, invmsk = {}};
               end
               pyr[i].orientation[o] = {ori = {}};
            end
         end

      elseif (pyr1[1].data ~= nil) then
         pyr[i] = { data = {} };
   
      elseif (pyr1[1].hData ~= nil) then
         pyr[i] = { hData = {}, vData = {} };
      end
   end

   if (#pyr1 == 0) or (pyr1 == nil) then
      pyr = torch.DoubleTensor(pyr2);
   else
      if (pyr1[1].orientation ~= nil) then
         for level = 1,#pyr1 do
            for ori = 1,#pyr1[level].orientation do
               pyr[level].orientation[ori].data = torch.DoubleTensor(pyr1[level].orientation[ori].data + pyr2[level].orientation[ori].data);
               if (pyr1[level].orientation[1].ori ~= nil) then
                  pyr[level].orientation[ori].ori = pyr1[level].orientation[ori].ori;
                  if (pyr1[level].orientation[1].invmsk ~= nil) then
                     pyr[level].orientation[ori].invmsk = pyr1[level].orientation[ori].invmsk;
                  end
               end
            end
         end
      
      elseif (pyr1[1].data ~= nil) then
         for level = 1,#pyr1 do
            pyr[level].data = torch.DoubleTensor(pyr1[level].data + pyr2[level].data);
         end
            
      elseif (pyr1[1].hData ~= nil) then
         pyr[level].hData = torch.DoubleTensor(pyr1[level].hData + pyr2[level].hData);
         pyr[level].vData = torch.DoubleTensor(pyr1[level].vData + pyr2[level].vData);
      else
         print("Error in pyramid summation");
      end
   end

   return pyr;
end

--
--function: Normalize CS Pyramids (2)
--
function normCSPyr2(csPyr1,csPyr2)
   local newPyr1 = {};
   local newPyr2 = {};
   for i = 1,#csPyr1 do
      newPyr1[i] = { data = {} }
      newPyr2[i] = { data = {} }
   end
   
   local temp;
   local norm;
   local scale;
   for level = 1,#csPyr1 do
      temp = sumPyr(csPyr1,csPyr2);
      norm = maxNormalizeLocalMax(temp[level].data,torch.DoubleTensor({0,10}));

      if (torch.max(temp[level].data) ~= 0) then
         scale = torch.max(norm) / torch.max(temp[level].data);
      else
         scale = 0;
      end
      
      newPyr1[level].data = csPyr1[level].data:clone() * scale;
      newPyr2[level].data = csPyr2[level].data:clone() * scale;
   end

   return newPyr1,newPyr2;
end

--
--function: Factorial
--
function fact (n)
      if n == 0 then
        return 1
      else
        return n * fact(n-1)
      end
end

--
--function: Make Von Mises
--
function makeVonMises(R0, theta0, dim1, dim2)
   local msk1 = torch.DoubleTensor(dim1:size()[1],dim2:size()[1]):zero();
   local msk2 = torch.DoubleTensor(dim1:size()[1],dim2:size()[1]):zero();  

   local sigma_r = R0/2;
   local B = R0;
   
   if (dim2[1] == torch.min(dim2)) then
      local dim2_temp = {};
      for i = 0,dim2:size()[1]-1 do
         dim2_temp[i + 1] = dim2[dim2:size()[1] - i];
      end
      dim1 = torch.DoubleTensor(1,dim1:size()[1]):copy(dim1)
      dim2 = torch.DoubleTensor(dim2_temp);
      dim2 = torch.DoubleTensor(dim2:size()[1],1):copy(dim2);
   else
      dim1 = torch.DoubleTensor(1,dim1:size()[1]):copy(dim1)
      dim2 = torch.DoubleTensor(dim2:size()[1],1):copy(dim2);
   end
   
   --make grid
   local X,Y;
   X,Y = meshGrid(dim1,dim2);
   
   --convert to polar coordinates
   local R = torch.sqrt(torch.pow(X,2) + torch.pow(Y,2));
   local theta = torch.atan2(Y,X);

   --make mask
   -----------
   -----------besseli
   msk1 = torch.cdiv(torch.exp(torch.mul(torch.cos(theta - (theta0)),B)),besseli(R-R0,torch.DoubleTensor(R:size()):zero(),0))
   msk1 = torch.div(msk1,torch.max(msk1));

   --msk1 = torch.cdiv(torch.exp(torch.mul(torch.cos(theta - theta0),B)),besseli2(R-R0))
   -----------
   -----------

   -----------
   -----------besseli
   msk2 = torch.cdiv(torch.exp(torch.mul(torch.cos(theta - (theta0 + math.pi)),B)),besseli(R-R0,torch.DoubleTensor(R:size()):zero(),0))
   msk2 = torch.div(msk2,torch.max(msk2));

   --msk2 = torch.cdiv(torch.exp(torch.mul(torch.cos(theta - (theta0 + math.pi)),B)),besseli2(R-R0))
   -----------
   -----------

   return msk1,msk2;
end

--
--function: Von Mises Pyramid
--
function vonMisesPyramid(map, vmPrs, fil_vals1, fil_vals2)
   local pyr1 = {};
   local pyr2 = {};
   local msk1 = {};
   local msk2 = {};
   for l = vmPrs.minLevel,vmPrs.maxLevel do
      pyr1[l] = { orientation = {} };
      pyr2[l] = { orientation = {} };
      msk1[l] = { orientation = {} };
      msk2[l] = { orientation = {} };
   end
   
   local temp_data;
   for level = vmPrs.minLevel,vmPrs.maxLevel do
      if(#map[level].data ~= 0) then
         temp_data = torch.DoubleTensor(map[level].data);
         temp_data = prePadImage(torch.DoubleTensor(temp_data:size()):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE,0);
         
         conv:forward(temp_data);
         for ori = 1,vmPrs.numOri do
            -------------------------------
            -------------------------------            
            pyr1[level].orientation[ori] = {data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_vals1[1][ori]]), ori = vmPrs.oris[ori] + (math.pi / 2)};
            -------------------------------
            -------------------------------
            pyr2[level].orientation[ori] = {data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_vals2[1][ori]]), ori = vmPrs.oris[ori] + (math.pi / 2)};
            -------------------------------
            -------------------------------
            msk1[level].orientation[ori] = {data = fil_vals2[1][ori], ori =  vmPrs.oris[ori] + (math.pi / 2)};
            msk2[level].orientation[ori] = {data = fil_vals1[1][ori], ori = vmPrs.oris[ori] + (math.pi / 2)};
         end
      else
         print('Map is empty at specified level.');
      end
   end
   
   return pyr1,msk1,pyr2,msk2;
end

--
--function: Von Mises Sum
--
function vonMisesSum(csPyr, vmPrs,vmfil_vals1,vmfil_vals2)
   local maxLevel = vmPrs.maxLevel;
   --local maxLevel = 5
   local map1 = {};
   local map2 = {};
   local vmPyr1;
   local msk1;
   local vmPyr2;
   local msk2;

   --create pyramid of center surround convoled with von Mises distribution
   vmPyr1, msk1, vmPyr2, msk2 = vonMisesPyramid(csPyr,vmPrs,vmfil_vals1,vmfil_vals2);   
   for level = 1, maxLevel do
      map1[level] = { orientation = {} };
      map2[level] = { orientation = {} };
      for ori = 1,vmPrs.numOri do
         map1[level].orientation[ori] = {data = {}};
         map2[level].orientation[ori] = {data = {}};
      end
   end
   
   --sum convolved output across different spatial scales
   for minL = 1,maxLevel do
      local mapLevel = minL;
      for l = minL,maxLevel do
         for ori = 1,vmPrs.numOri do
            if (l == minL) then
               map1[minL].orientation[ori].data = torch.DoubleTensor(vmPyr1[mapLevel].orientation[ori].data:size()):zero();
               map2[minL].orientation[ori].data = torch.DoubleTensor(vmPyr2[mapLevel].orientation[ori].data:size()):zero();
            end
            local temp;
            temp = image.scale(vmPyr1[l].orientation[ori].data[1],vmPyr1[mapLevel].orientation[ori].data:size()[3],vmPyr1[mapLevel].orientation[ori].data:size()[2],'simple');
            map1[minL].orientation[ori].data:add(math.pow(0.5,l-1),temp)

            temp = image.scale(vmPyr2[l].orientation[ori].data[1],vmPyr1[mapLevel].orientation[ori].data:size()[3],vmPyr1[mapLevel].orientation[ori].data:size()[2],'simple');
            map2[minL].orientation[ori].data:add(math.pow(0.5,l-1),temp)
         end
      end   
   end
   
   return map1,msk1,map2,msk2;
end

--
--function: Border Pyramid
--
function borderPyramid(csPyrL,csPyrD,cPyr,params,VM_fil_vals1,VM_fil_vals2)
   --timerr = torch.Timer();      
   local bPrs = params.bPrs;
   local vmPrs = params.vmPrs;

   local bPyr1 = {};
   local bPyr2 = {};
   local bPyr3 = {};
   local bPyr4 = {};
   for level = bPrs.minLevel, bPrs.maxLevel do
      bPyr1[level] = {orientation = {}};
      bPyr2[level] = {orientation = {}};
      bPyr3[level] = {orientation = {}};
      bPyr4[level] = {orientation = {}};
      for ori = 1,bPrs.numOri do
         bPyr1[level].orientation[ori] = {data = {}};
         bPyr2[level].orientation[ori] = {data = {}};
         bPyr3[level].orientation[ori] = {data = {}};
         bPyr4[level].orientation[ori] = {data = {}};
      end
   end

   --convolve center surround with von Mises distribution and, for every orientation, sum across all spatial scales greater or equal to scale 1
   local vmL1,msk1,vmL2,msk2;
   local vmD1, csmsk1,vmD2,csmsk2;
   
   --timerr = torch.Timer();
   vmL1, msk1, vmL2, msk2 = vonMisesSum(csPyrL,vmPrs,VM_fil_vals1,VM_fil_vals2);
   --print('Von Mises Sum 1 = ' .. timerr:time().real);         
   --timerr = torch.Timer();   
   vmD1, csmsk1, vmD2, csmsk2 = vonMisesSum(csPyrD, vmPrs,VM_fil_vals1,VM_fil_vals2);
   --print('Von Mises Sum 2 = ' .. timerr:time().real);
   

   local bpyr1_temp;
   local bpyr2_temp;

   --calculate border ownership and grouping responses
   for level = bPrs.minLevel,bPrs.maxLevel do
      for ori_cnt = 1,bPrs.numOri do
         --create border ownership for light objects (on center CS results)
         bpyr1_temp = torch.cmul(cPyr[level].orientation[ori_cnt].data,torch.add(torch.mul(vmL1[level].orientation[ori_cnt].data,bPrs.alpha) - torch.mul(vmD2[level].orientation[ori_cnt].data,bPrs.CSw),1));
         bpyr1_temp[torch.lt(bpyr1_temp,0)]=0;
         bPyr1[level].orientation[ori_cnt] = {data = bpyr1_temp:clone(), ori = msk1[level].orientation[ori_cnt].ori, invmsk = msk1[level].orientation[ori_cnt].data};
         
         bpyr2_temp = torch.cmul(cPyr[level].orientation[ori_cnt].data,torch.add(torch.mul(vmL2[level].orientation[ori_cnt].data,bPrs.alpha) - torch.mul(vmD1[level].orientation[ori_cnt].data,bPrs.CSw),1));
         bpyr2_temp[torch.lt(bpyr2_temp,0)]=0;
         bPyr2[level].orientation[ori_cnt] = {data = bpyr2_temp:clone(), ori = msk2[level].orientation[ori_cnt].ori + math.pi, invmsk = msk2[level].orientation[ori_cnt].data};
         
         --create border ownership for dark objects (off center cs results)
         bpyr3_temp = torch.cmul(cPyr[level].orientation[ori_cnt].data,torch.add(torch.mul(vmD1[level].orientation[ori_cnt].data,bPrs.alpha) - torch.mul(vmL2[level].orientation[ori_cnt].data,bPrs.CSw),1));
         bpyr3_temp[torch.lt(bpyr3_temp,0)]=0;
         bPyr3[level].orientation[ori_cnt] = {data = bpyr3_temp:clone(), ori = msk1[level].orientation[ori_cnt].ori, invmsk = msk1[level].orientation[ori_cnt].data};
         
         bpyr4_temp = torch.cmul(cPyr[level].orientation[ori_cnt].data,torch.add(torch.mul(vmD2[level].orientation[ori_cnt].data,bPrs.alpha) - torch.mul(vmL1[level].orientation[ori_cnt].data,bPrs.CSw),1));
         bpyr4_temp[torch.lt(bpyr4_temp,0)]=0;
         bPyr4[level].orientation[ori_cnt] = {data = bpyr4_temp:clone(), ori = msk2[level].orientation[ori_cnt].ori + math.pi, invmsk = msk2[level].orientation[ori_cnt].data};
      end
   end
   --print('border pyramid FINAL = ' .. timerr:time().real);

   return bPyr1,bPyr2,bPyr3,bPyr4;
end

--
--function: Make Border Ownership
--
function makeBorderOwnership(im_channels,params,even_fil_vals,odd_fil_vals,cs_fil_val,vm1_fil_vals,vm2_fil_vals)
   local map;
   local imPyr;
   local EPyr;
   local OPyr;
   local cPyr;
   local csPyr;
   local csPyrL,csPyrD;
   local bPyr1_1,bPyr2_1,bPyr1_2,bPyr2_2;
   
   local b1Pyr = {};
   local b2Pyr = {};
 
   for level = 1,#im_channels do
      b1Pyr[level] = {subtype = {},subname = {}, type = {}};
      b2Pyr[level] = {subtype = {},subname = {}, type = {}};
   end
   

   --EXTRACT EDGES
   for m = 1,#im_channels do

      for sub = 1,#im_channels[m].subtype do
         map = torch.DoubleTensor(im_channels[m].subtype[sub].data):clone();
         imPyr = makePyramid(map,params);
         ------------------
         --Edge Detection--
         ------------------
         --timerr = torch.Timer();
         EPyr = edgeEvenPyramid(imPyr,params,even_fil_vals);
         --print('MakeEvenEdge = ' .. timerr:time().real);         
         --timerr = torch.Timer();
         OPyr, o = edgeOddPyramid(imPyr,params,odd_fil_vals);
         --print('MakeOddEdge = ' .. timerr:time().real);      
         --timerr = torch.Timer();   
         cPyr = makeComplexEdge(EPyr,OPyr,cs_fil_val);
         --print('MakeComplexEdge = ' .. timerr:time().real);         

         ----------------------
         --Make Image Pyramid--
         ----------------------
         if(im_channels[m].subtype[sub].type == "Orientation") then
            --timerr = torch.Timer();   
            csPyr = gaborPyramid(imPyr,im_channels[m].subtype[sub].ori,params,im_channels[m].subtype[sub].filval);
            --print('Make Gabor Pyramid (orientation) = ' .. timerr:time().real);
         else
            --timerr = torch.Timer();   
            csPyr = csPyramid(imPyr,params,cs_fil_val);
            --print('Make center surround pyramid = ' .. timerr:time().real);
         end
         
         --timerr = torch.Timer();   
         csPyrL,csPyrD = separatePyr(csPyr);
         --print('separate pyramid = ' .. timerr:time().real);
         --timerr = torch.Timer();
         csPyrL,csPyrD = normCSPyr2(csPyrL,csPyrD);
         --print('normalize cs pyramid = ' .. timerr:time().real);
         -----------------------------------------------
         --Generate Border Ownership and Grouping Maps--
         -----------------------------------------------
         --timerr = torch.Timer();
         bPyr1_1, bPyr2_1, bPyr1_2, bPyr2_2 = borderPyramid(csPyrL,csPyrD,cPyr,params,vm1_fil_vals,vm2_fil_vals);
         --print('border pyramid final  = ' .. timerr:time().real);
         b1Pyr[m].subtype[sub] = sumPyr(bPyr1_1,bPyr1_2);
         b2Pyr[m].subtype[sub] = sumPyr(bPyr2_1,bPyr2_2);

         if (im_channels[m].subtype[sub].type == "Orientation") then
            b1Pyr[m].subname[sub] = rad2deg(im_channels[m].subtype[sub].ori) .. " deg";
            b2Pyr[m].subname[sub] = rad2deg(im_channels[m].subtype[sub].ori) .. " deg";
         else
            b1Pyr[m].subname[sub] = im_channels[m].subtype[sub].type;
            b2Pyr[m].subname[sub] = im_channels[m].subtype[sub].type;
         end
      end
      b1Pyr[m].type = im_channels[m].type;
      b2Pyr[m].type = im_channels[m].type;
   end
   
   return b1Pyr,b2Pyr;
end

--
--function: Grouping Pryamid Max Difference
--
function groupingPyramidMaxDiff(bpyrr1,bpyrr2,params)
   local bPrs = params.bPrs;
   local giPrs = params.giPrs;
   local w = giPrs.w_sameChannel;
   
   local gPyr1 = {};
   local gPyr2 = {};
   
   for l = bPrs.minLevel,bPrs.maxLevel do
      gPyr1[l] = {orientation = {}};
      gPyr2[l] = {orientation = {}};
      for o = 1,bPrs.numOri do
         gPyr1[l].orientation[o] = {data = {}};
         gPyr2[l].orientation[o] = {data = {}};
      end
   end
   
   local bTemp1;
   local m1,m_ind1;
   local m_i1;
   local invmsk1,invmsk2;
   local temp_data1_1,temp_data1_2,temp_data2_1,temp_data2_2;
   local out1_1,out1_2,out2_1,out2_2;
   local final1,final2;
   local outputs;
   
   --calculate border ownership and grouping responses
   for level = bPrs.minLevel, bPrs.maxLevel do
      bTemp1 = torch.DoubleTensor(bPrs.numOri,bpyrr1[level].orientation[1].data:size(2),bpyrr1[level].orientation[1].data:size(3)):zero();
      for ori = 1,bPrs.numOri do
         bTemp1[ori][{{},{}}] = torch.squeeze(torch.abs(bpyrr1[level].orientation[ori].data - bpyrr2[level].orientation[ori].data));
      end
      m1, m_ind1 = torch.max(bTemp1,1);
      m_ind1 = m_ind1:type('torch.DoubleTensor');
      
      for ori = 1,bPrs.numOri do
         m_i1 = torch.squeeze(m_ind1:clone());
         m_i1[torch.ne(m_i1,ori)] = 0;
         m_i1[torch.eq(m_i1,ori)] = 1;
         
         invmsk1 = bpyrr1[level].orientation[ori].invmsk;
         invmsk2 = bpyrr2[level].orientation[ori].invmsk;
         
         b1p = torch.cmul(m_i1,bpyrr1[level].orientation[ori].data - bpyrr2[level].orientation[ori].data);
         b1n = b1p:clone() * -1;
         b1p[torch.lt(b1p,0)] = 0;
         b1p[torch.ne(b1p,0)] = 1;
         b1n[torch.lt(b1n,0)] = 0;
         b1n[torch.ne(b1n,0)] = 1;
         
         temp_data1_1 = torch.cmul(bpyrr1[level].orientation[ori].data,b1p);         
         temp_data1_1 = prePadImage(torch.DoubleTensor(1,temp_data1_1:size()[2],temp_data1_1:size()[3]):copy(temp_data1_1),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
         conv:forward(temp_data1_1);
         outputs = conv.output;
         out1_1 = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[invmsk1]);
         
         temp_data1_2 = torch.cmul(bpyrr2[level].orientation[ori].data,b1p * w);      
         temp_data1_2 = prePadImage(torch.DoubleTensor(1,temp_data1_2:size()[2],temp_data1_2:size()[3]):copy(temp_data1_2),MAX_FILTER_SIZE,MAX_FILTER_SIZE, torch.max(temp_data1_2));
         conv:forward(temp_data1_2);
         outputs = conv.output;
         out1_2 = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[invmsk1]);
         
         temp_data2_1 = torch.cmul(bpyrr2[level].orientation[ori].data,b1n);
         temp_data2_1 = prePadImage(torch.DoubleTensor(1,temp_data2_1:size()[2],temp_data2_1:size()[3]):copy(temp_data2_1),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
         conv:forward(temp_data2_1);
         outputs = conv.output;
         out2_1 = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[invmsk2]);
         
         temp_data2_2 = torch.cmul(bpyrr1[level].orientation[ori].data,b1n * w);
         temp_data2_2 = prePadImage(torch.DoubleTensor(1,temp_data2_2:size()[2],temp_data2_2:size()[3]):copy(temp_data2_2),MAX_FILTER_SIZE,MAX_FILTER_SIZE, torch.max(temp_data2_2));
         conv:forward(temp_data2_2);
         outputs = conv.output;
         out2_2 = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[invmsk2]);
         
         final1 = out1_1 - out1_2;
         final2 = out2_1 - out2_2;
         
         final1[torch.lt(final1,0)] = 0;
         final2[torch.lt(final2,0)] = 0;
         
         gPyr1[level].orientation[ori].data = final1:clone();
         gPyr2[level].orientation[ori].data = final2:clone();
      end

   end
   
   return gPyr1, gPyr2;
end

--
--function Merge Level
--
function mergeLevel(pyrr)
   local newPyr = {};
   local temp = {};
   
   for l = 1,#pyrr do
      newPyr[l] = {data = {}};
   end

   for level = 1,#pyrr do
      if (pyrr[1].orientation[1] ~= nil) then
         temp = torch.DoubleTensor(pyrr[level].orientation[1].data:size()):zero();
         for ori =1,#pyrr[level].orientation do
            temp = torch.add(temp,pyrr[level].orientation[ori].data);
         end
         newPyr[level].data = temp:clone();
      end
   end
   
   return newPyr;
end

--
--function: Make Grouping
--
function makeGrouping(b1Pyrr,b2Pyrr,params)
   
   local gPyr = {};
   local gPyr1_1 = {};
   local gPyr2_1 = {};
   local g11 = {};
   local g21 = {};
   
   for m = 1,#b1Pyrr do
      gPyr1_1[m] = {subtype = {}};
      gPyr2_1[m] = {subtype = {}};
      gPyr[m] = {type = {}, subtype = {}};
   end

   for m = 1,#b1Pyrr do
      --print("\nAssigning Grouping on " .. b1Pyrr[m].type .. " channel:\n");
      for sub = 1,#b1Pyrr[m].subtype do
         --print("Subtype " .. sub .. " of " .. #b1Pyrr[m].subtype);
         --print(b1Pyrr[m].subname[sub] .. "\n");
         
         gPyr1_1[m].subtype[sub], gPyr2_1[m].subtype[sub] = groupingPyramidMaxDiff(b1Pyrr[m].subtype[sub],b2Pyrr[m].subtype[sub],params);    
      timerr = torch.Timer();

         g11 = mergeLevel(gPyr1_1[m].subtype[sub]);
         g21 = mergeLevel(gPyr2_1[m].subtype[sub]);
         print('grouping pyramid MAX DIFF  = ' .. timerr:time().real);

         gPyr[m].subtype[sub] = sumPyr(g11,g21);
      end
      gPyr[m].type = b1Pyrr[m].type;
   end
   
   return gPyr;
end

--
--function: Itti et. al. Normalization
--
function ittiNorm(gPyrr,collapseLevel)
   local CM = {};
   for mm = 1,#gPyrr do
      CM[mm] = {data = {}};
   end
   
   local FM;
   local CML;
   local FML;

   for m = 1,#gPyrr do
      FM = {};
      if(gPyrr[m].type ~= "Orientation") then
         for l = 1,#gPyrr[m].subtype[1] do
            FM[l] = { data = {}}
         end

         for level = 1,#gPyrr[m].subtype[1] do
            FM[level].data = torch.DoubleTensor(gPyrr[m].subtype[1][level].data:size(2),gPyrr[m].subtype[1][level].data:size(3)):zero();
            for sub = 1,#gPyrr[m].subtype do
               temp_normalized = maxNormalizeLocalMax(gPyrr[m].subtype[sub][level].data:clone(),torch.DoubleTensor({0,10}));
               FM[level].data = torch.add(FM[level].data:clone(),temp_normalized:clone()):clone();
            end
         end

         CML = {};
         for l = 1,#FM do
            CML[l] = {data = FM[l].data:clone()};
         end

         CM[m].data = torch.DoubleTensor(FM[collapseLevel].data:size()):zero();
         for l = 1,#CML do
            temp_resized = image.scale(CML[l].data, FM[collapseLevel].data:size(2), FM[collapseLevel].data:size(1),'simple');
            CM[m].data = torch.add(CM[m].data,temp_resized);
         end

      elseif (gPyrr[m].type == "Orientation") then
         FM = gPyrr[m].subtype;
         CM[m].data = torch.DoubleTensor(FM[1][collapseLevel].data:size()):zero();
         CML = {};
         FML = {};
         
         for sub = 1,#FM do
            temp = {};
            for i = 1,#FM[m] do
               temp[i] = {data = i + sub};
            end
            FML[sub] = temp;
         end

         for sub = 1,#FM do
            CML[sub] = {data = torch.DoubleTensor(FM[1][collapseLevel].data:size()):zero()};
            
            for l = 1,#FM[m] do
               temp_norm = maxNormalizeLocalMax(FM[sub][l].data,torch.DoubleTensor({0,10}));
               FML[sub][l] = {data = temp_norm};
               temp_resize = image.scale(FML[sub][l].data,FM[1][collapseLevel].data:size(3), FM[1][collapseLevel].data:size(2),'simple');
               CML[sub].data = torch.add(CML[sub].data,temp_resize);
            end
            CM[m].data = torch.add(CM[m].data,CML[sub].data);
         end

      else
         print("Please ensure algorithm operates on known feature types");
      end
   end
   
   local h = {data = torch.DoubleTensor(CM[1].data:size()):zero()};
   local temp_normd;

   for m = 1,#CM do
      temp_normd = maxNormalizeLocalMax(CM[m].data,torch.DoubleTensor({0,10}));
      temp_normd = torch.mul(temp_normd,1/3);
      h.data = torch.add(h.data,temp_normd);
   end
   
   return h;
   
end
