--------------------------------
--Proto-Object Saliency README--
--------------------------------

--**CONVOLUTION/CORRELATION COMPUTATION REQUIRED

--
--1) CREATE CHANNELS
--
--Separates raw image into 3 channels (intensity, color, orientation)
--Intensity - 1 gray-scaled version of image (1 channel)
--Color - 1 R-G opponency, 1 G-R opponency, 1 Y-B opponency, 1 B-Y opponency (4 channels)
--Orientation - 4 different orientations (4 channels)

--
--2) CREATE PYRAMID
--
--Create pyramid at max 10 levels
--Decreasing scales

--
--3) EXTRACT EDGES **
--
--For each channel, extracts edges using even and odd 2D Gabor filters
--Performs operation sequentially on each channel and each scale
--Performs operation on 4 different orientations

--
--4) COMPUTE COMPLEX CELL RESPONSE
--
--Using even and odd gabor filter responses, computes complex cell responses
--Computes for each of 4 orientations

--
--5) COMPUTE CENTER SURROUND **
--
--Compute ON-center and OFF-center center-surround responses for each channel (independently)
--For Orientation channel, compute ON/OFF center-surround responses in appropriate orientation/direction

--
--6) COMPUTE VON MISES RESPONSE **
--
--The von mises distribution kernel is computed at each orientation on the center-surround responses
--This is used to map activity of center-surround back to complex cell edge responses
--

--
--7) COMPUTE BORDER OWNERSHIP
--
--Using Complex edge responses and von mises responses to center-surround responses, the border ownership responses are computed
--The border ownership is computed for dark objects on light backgrounds and light objects on dark backgrounds (independently)
--Each are summed (at each level and each orientation)
--Border ownerships for theta and theta + pi are computed (two antagonistic pairs of border ownership responses for dark obj on light background and vice-versa = total of 4 pyramids of border ownership responses)

--
--8) GROUPING MECHANISM **
--
--Computes max operator on most active border ownership response (out of each orientation)
--Computes correlation with von mises kernel to compute grouping activity
--Computes subtraction operator for final grouping activity

--
--9) COMPUTE ITTI ET. AL, (1998) NORMALIZATION
--
--Normalizes and sums over each subtype for each channel
--Normalizes and sums over each channel
--Uses same normalization process for both
