L = 1;
h = .05;
l = .2;
nw = .008;
lc = L/5;
lc1 = .025;
Point(1) = {0, 0, 0, l};
//+
Point(2) = {L, 0, 0, l};
//+
Point(3) = {L, L , 0, l};
//+
Point(4) = {0, L, 0, l};
Point(5) = {0, .5*(L+nw), 0, l};
Point(6) = {.5*L, .5*(L+nw), 0, nw};
Point(7) = {.5*L, .5*(L-nw), 0, nw};
Point(8) = {0, .5*(L-nw), 0, l};

//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8,1};

//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7,8};
//+
Plane Surface(1) = {1};
//+
Physical Curve(103) = {3};
//+
Physical Curve(104) = {2,  4, 5, 6,7,8};
//+
Physical Curve(203) = {1};
//+
Physical Surface(2000) = {1};
Field[1] = Box;
//+
Field[1].Thickness = lc;
//+
Field[1].VIn = lc1/6;
//+
Field[1].VOut = 0.1;
//+
Field[1].XMax = 1;
//+
Field[1].XMin = 0.3;
//+
Field[1].YMax = .5*(L+lc);
//+
Field[1].YMin = .5*(L-lc);
//+
Background Field = 1;
