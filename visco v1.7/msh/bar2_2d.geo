// Gmsh project created on Thu Aug 11 11:07:54 2022

L = 5; 
h = 1;
//mesh size dx
dx = L/40;
//+
Point(1) = {0, 0, 0, dx};
//+
Point(2) = {5, 0, 0, dx};
//+
Point(3) = {5, 1, 0, dx};
//+
Point(4) = {0, 1, 0, dx};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Point(20) = {1};
//+
Physical Point(21) = {2};
//+
Physical Point(22) = {3};
//+
Physical Point(23) = {4};
//+
Physical Curve(5) = {1};
//+
Physical Curve(6) = {2};
//+
Physical Curve(7) = {3};
//+
Physical Curve(8) = {4};
//+
Physical Surface(9) = {1};
