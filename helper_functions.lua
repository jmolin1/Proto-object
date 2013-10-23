---------------------------
--MISCELLANEOUS FUNCTIONS--
---------------------------

--
--function: Write to File
--
function writeToFile(data,filename)
   local file = torch.DiskFile('/home/jamal/Dropbox/ProtoObject/ProtoObjectStatic/' .. filename, 'w')
   file:writeObject(data:size())
   file:writeObject(data)
   file:close()
   print('Data written to file: ' .. filename);
end

--
--function: Degrees to Radians
--
function deg2rad(angleInDegrees)
   local angleInRadians = angleInDegrees * (math.pi/180);
   return angleInRadians;
end

--
--function: Radians to Degrees
--
function rad2deg(angleInRadians)
   local angleInDegrees = angleInRadians  * (180/math.pi);
   return angleInDegrees;
end

--
--function: Safe Divide
--
function safeDivide(arg1, arg2)
   local ze = torch.eq(arg2,0):clone();
   arg2[ze] = 1;
   local result = torch.cdiv(arg1,arg2);
   result[ze] = 0
   return result;
end

--
--function Mesh Grid
--
function meshGrid(x_array,y_array) 
   --Get size
   local xsize = torch.numel(x_array);
   local ysize = torch.numel(y_array);

   --Meshgrid
   local x = torch.expand(x_array,ysize,xsize);
   local y = torch.expand(y_array,ysize,xsize);
   return x,y;
end

--
--function: Clamp Data
--
function clamp(data,bottom,top)
   if(bottom ~= nil) then
      data[torch.lt(data,bottom)] = bottom;
   end
   
   if(top ~= nil) then
       data[torch.gt(data,top)] = top;
   end
   
   return data;
end

--
--function: Modified Besseli function
--
function besseli(Z,finalZ,k)
   if (k < 15) then
      local final_divisor = math.pow(fact(k),2);
      local finalZ = finalZ + torch.div(torch.pow(torch.div(torch.pow(Z,2),4),k),final_divisor);
      return besseli(Z,finalZ,k+1);
   else
      return finalZ;
   end
end

--
--function: Modified Besseli function
--
function besseli2(Z)
   local finalZ = torch.DoubleTensor(Z:size()):zero();
   for k = 0,15 do
      divisor = math.pow(fact(k),2);
      finalZ = finalZ + torch.div(torch.pow(torch.div(torch.pow(Z,2),4),k),divisor);
   end
   
   return torch.DoubleTensor(finalZ);
end

--
--function: Calc Sigma
--
function calcSigma(r,x)
   local sigma1 = (r^2) / (4 * math.log(x)) * (1-(1/(x^2)));
   sigma1 = math.sqrt(sigma1);
   local sigma2 = x * sigma1;
   
   return sigma1,sigma2;
end
