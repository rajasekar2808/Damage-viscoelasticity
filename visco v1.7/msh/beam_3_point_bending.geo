// Gmsh project created on Sat Jul 16 22:44:54 2022

//crack width 2w
w = .6;
//crack  length a
a = 8;

// mesh size at points
h = 2*w; 
h1 = 5;

//loading line
le = 2;


//+
Point(1) = {0, 0, 0, 1.3*h1};
//+
Point(2) = {10-.5*le, 0, 0, .3*h1};
//+
Point(3) = {10+.5*le,0,0,.3*h1};
//+
Point(4) = {130-w, 0, 0, .5*h1};
//+
Point(5) = {130-w,a,0,h};
//+
Point(6) = {130+w,a,0,h};
//+
Point(7) = {130+w,0,0,.5*h1};
//+
Point(8) = {10+240-.5*le,0,0,.3*h1};
//+
Point(9) = {10+240+.5*le,0,0,.3*h1};
//+
Point(10) = {260,0,0,1.3*h1};
//+
Point(11) = {260,40.,0,1.5*h1};
//+
Point(12) = {130+le,40,0,.3*h1};
//+
Point(13) = {130-le,40,0,.3*h1};
//+
Point(14) = {0,40,0,1.5*h1};

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
Line(12) = {12,13};
//+
Line(13) = {13,14};
//+
Line(14) = {14,1};
//+



Physical Curve(101) = {2};
//+
Physical Curve(102) = {8};
//+ 
Physical Curve(103) = {12};
//+
Physical Curve(1) = {1, 3, 4, 5, 6, 7, 9, 10, 11,13,14};
//+


Curve Loop(1) = {1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14};
//+
Plane Surface(1) = {1};
//+
Physical Surface(1000) = {1};
//+
Field[1] = Box;
//+
Field[1].Thickness = 30;
//+
Field[1].VIn = .75*w;
//+
Field[1].VOut = 10;
//+
Field[1].XMax = 130+15;
//+
Field[1].XMin = 130-15;
//+
Field[1].YMax = 40;
//+
Field[1].YMin = a/4;
//+
Background Field = 1;
