//units :  millimeter
// hc = 1.; // _r0 edge size along the crack line
//hc= 0.5;    // _r1 edge size along the crack line
//hc = 0.25; // _r2 edge size along the crack line
hc = 0.125; // _r3 edge size along the crack line

hf = 4.*hc; // edge size along the other lines

L1= 100.;
L2 = 12.;
L3 = 20.;
L4 = 24.;
H1 = 70.;
H2 = 90.;
H3 = 24.;
H4 = 5.;
R  = 4.;

Point(1)  = {L4,       0.,   0, hc};
Point(2)  = {L3,    H4/2.,   0, hf};
Point(3)  = {0.,    H4/2.,   0, hf};
Point(4)  = {0.,    H1/2.,   0, hf};
Point(5)  = {L1,    H2/2.,   0, hf};
Point(6)  = {L1,   -H2/2.,   0, hf};
Point(7)  = {0.,   -H1/2.,   0, hf};
Point(8)  = {0.,   -H4/2.,   0, hf};
Point(9)  = {L3,   -H4/2.,   0, hf};
Point(10) = {L1,       0.,   0, hc};

Point(11) = {L2,   -H3/2.,   0, hf};
Point(12) = {L2,    H3/2.,   0, hf};

Point(13) = {L2+R, -H3/2.,   0, hf};
Point(14) = {L2,   -H3/2.+R, 0, hf};
Point(15) = {L2-R, -H3/2.,   0, hf};
Point(16) = {L2,   -H3/2.-R, 0, hf};

Point(17) = {L2+R,  H3/2.,   0, hf};
Point(18) = {L2,    H3/2.+R, 0, hf};
Point(19) = {L2-R,  H3/2.,   0, hf};
Point(20) = {L2,    H3/2.-R, 0, hf};

//+
Line(1) = {8, 7}; 
//+
Line(2) = {7, 6};
//+

//+
Line(5) = {5, 4};
//+
Line(6) = {4, 3};
//+
Line(7) = {3, 2};
//+
Line(8) = {2, 1};
//+
Line(9) = {1, 9};
//+
Line(10) = {9, 8};
//+
Line(11) = {1, 10};
//+
Circle(12) = {20, 12, 17};
//+
Circle(13) = {17, 12, 18};
//+
Circle(14) = {18, 12, 19};
//+
Circle(15) = {19, 12, 20};
//+
Circle(16) = {16, 11, 13};
//+
Circle(17) = {13, 11, 14};
//+
Circle(18) = {14, 11, 15};
//+
Circle(19) = {15, 11, 16};
Point(21) = {20, -6, 0, 1.0};
//+
Point(22) = {20, 6, 0, 1.0};
//+
Point(23) = {100, 6, 0, 1.0};
//+
Point(24) = {100, -6, 0, 1.0};
//+
Line(20) = {22, 2};
//+
Line(21) = {9, 21};
//+
Line(22) = {21, 24};
//+
Line(23) = {22, 23};
//+
Line(24) = {5, 23};
//+
Line(25) = {23, 10};
//+
Line(26) = {10, 24};
//+
Line(27) = {24, 6};
//+
Curve Loop(1) = {1, 2, -27, -22, -21, 10};
//+
Curve Loop(2) = {17, 18, 19, 16};
//+
Plane Surface(1) = {1, 2};
//+
Curve Loop(3) = {6, 7, -20, 23, -24, 5};
//+
Curve Loop(4) = {13, 14, 15, 12};
//+
Plane Surface(2) = {3, 4};
//+
Curve Loop(5) = {20, 8, 9, 21, 22, -26, -25, -23};
//+
Plane Surface(3) = {5};
//+
Physical Curve(203) = {14, 13};
//+
Physical Curve(103) = {19, 16};
//+
Physical Curve(400) = {20, 8, 9, 21, 22, 26, 25, 23};
//+
Physical Surface(1000) = {3};
//+
Physical Surface(2000) = {1};
//+
Physical Surface(2001) = {2};
//+
Physical Curve(500) = {10, 1, 2, 27, 9, 8, 7, 25, 26, 24, 5, 6};
