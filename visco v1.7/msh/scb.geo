// Gmsh project created on Sat Feb 26 12:21:56 2022
//+

//Radius and length between supports
R = 75;
s = 120;

// Notch length and width (mm)
a = 10;   
w = .35;

// 2*l_c
lc = 20;  

// mesh size at selected points
m1 = 8;
m2 = 8;

//loading length
le = 2;
theta = le/R;
//pi = 3.14159265358979323846;
//angle = theta*pi/180;

Point(1) = {0, 0, 0, m2};
//+
Point(2) = {R-.5*s-.5*le, 0, 0, le/8};
//+
Point(3) = {R-.5*s+.5*le, 0, 0, le/8};
//
Point(4) = {R-w/2, 0, 0, m1};
//+
Point(5) = {R-w/2, a, 0, 2*w};
//+
Point(6) = {R+w/2, a, 0, 2*w};
//+
Point(7) = {R+w/2, 0, 0, m1};
//+
Point(8) = {R+.5*s-.5*le, 0, 0,  le/8};
//+
Point(9) = {R+.5*s+.5*le, 0, 0,  le/8};
//+
Point(10) = {2*R, 0, 0, m2};
//+
Point(11) = {R+R*Sin(theta), R*Cos(theta), 0, le/4};
//+
Point(12) = {R-R*Sin(theta), R*Cos(theta), 0, le/4};
//
Point(13) = {R, 0, 0, 1.0};


//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 9};
//+
Line(9) = {9, 10};
//+
Circle(10) = {10, 13, 11};
//+
Circle(11) = {11, 13, 12};
//+
Circle(12) = {12, 13, 1};

//+
Curve Loop(1) = {1, 2,3,4,5,6,7,8,9,10,11,12};
//+
Plane Surface(1) = {1};
//+
Physical Curve(101) = {2};
//+
Physical Curve(102) = {8};
//+
Physical Curve(103) = {11};
//+
Physical Curve(1) = {1,3,4,5,6,7,9,10,12};
//+
Physical Surface(28) = {1};
//+


Field[1] = Box;
//+
Field[1].Thickness = lc;
//+
Field[1].VIn = w;
//+
Field[1].VOut = 8;
//+
Field[1].XMax = R+lc/2;
//+
Field[1].XMin = R-lc/2;
//+
Field[1].YMin = a/4;
Field[1].YMax = R;
//+
Background Field = 1;
