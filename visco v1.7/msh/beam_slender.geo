// Gmsh project created on Thu Aug 11 11:21:03 2022

L = 800;
h = 40;
// mesh size dx
dx = 40/20;
//+
Point(1) = {0, 0, 0, dx};
//+
Point(2) = {L, 0, 0, dx};
//+
Point(3) = {L, h, 0, dx};
//+
Point(4) = {L/2, h, 0, dx};
//+
Point(5) = {0,h,0,dx};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 1};
//+
Curve Loop(1) = {1, 2, 3, 4,5};
//+
Plane Surface(1) = {1};
//+
Physical Point(12) = {1};
//+
Physical Point(13) = {2};
//+
Physical Point(14) = {4};
//+
Physical Curve(18) = {1};
//+
Physical Curve(19) = {2};
//+
Physical Curve(20) = {3};
//+
Physical Curve(21) = {4};
//+
Physical Curve(22) = {5};
//+
Physical Surface(9) = {1};
