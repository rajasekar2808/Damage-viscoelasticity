// Gmsh project created on Sat Jul 16 22:44:54 2022

//crack width 2w
w = .7;
//crack  length a
a = 19;

// mesh size at points
h = 2*w; 
h1 = 10;

// lenght of loading line
le = 2;

//+
Point(1) = {0, 0, 0, 1.3*h1};
//+
Point(2) = {26-.5*le, 0, 0, .3*le};
//+
Point(3) = {26+.5*le, 0, 0, .3*le};
//+
Point(4) = {26+97-w, 0, 0, .5*h1};
//+
Point(5) = {26+97-w,a,0,h};
//+
Point(6) = {26+97+w,a,0,h};
//+
Point(7) = {26+97+w,0,0,.5*h1};
//+
Point(8) = {26+162+162-.5*le,0,0,.3*le};
//+
Point(9) = {26+162+162+.5*le,0,0,.3*le};
//+
Point(10) = {26+162+162+26,0,0,1.3*h1};
//+
Point(11) = {26+162+162+26,100.,0,1.5*h1};
//+
Point(12) = {26+162+le,100.,0,.35*le};
//+
Point(13) = {26+162-le,100.,0,.35*le};
//+
Point(14) = {0,100,0,1.5*h1};

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
Line(10) = {10, 11};
//+
Line(11) = {11,12};
//+
Line(12) = {12, 13};
//+
Line(13) = {13, 14};
//+
Line(14) = {14,1};
//+



Physical Curve(101) = {2};
//+
Physical Curve(102) = {8};
//+ 
Physical Curve(103) = {12};
//+
Physical Curve(1) = {1,  3, 4, 5, 6, 7, 9, 10, 11, 13, 14};
//+


Curve Loop(1) = {1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14};
//+
Plane Surface(1) = {1};
//+
Physical Surface(1000) = {1};
//+
Field[1] = Box;
//+
Field[1].Thickness = 80;
//+
Field[1].VIn = 6*w;
//+
Field[1].VOut = 25;
//+
Field[1].XMax = 26+97+60;
//+
Field[1].XMin = 26+97-20;
//+
Field[1].YMax = 100;
//+
Field[1].YMin = a/4;
//+
Background Field = 1;
